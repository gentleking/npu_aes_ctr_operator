
#include "aes_ctr_custom_tiling.h" // 包含TilingData定义
#include "register/op_def_registry.h"
#include "register/operator.h"
#include "register/infer_axis_slice_registry.h"
#include "register/op_tiling_registry.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/utils/type_utils.h"

// 定义命名空间 ge 和 ops
namespace ge {
    // 函数前向声明
    static graphStatus InferShapeAesCtrCustom(Operator& op);
    static graphStatus InferDataTypeAesCtrCustom(Operator& op);
}

namespace optiling {

// AES分组大小（单位：byte）
const uint32_t AES_BLOCK_BYTES_CONST = 16; // AES块大小固定为16字节

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    // 1. 获取输入张量描述
    const gert::Tensor* plaintext_tensor = context->GetInput(0);
    if (plaintext_tensor == nullptr) {
        // printf("[ERROR] TilingFunc: Input plaintext_tensor is nullptr.\n");
        ERROR_LOG("Input plaintext_tensor is nullptr.");
        return ge::GRAPH_FAILED;
    }
    // 获取形状大小，对于DT_UINT8，GetShapeSize()直接返回字节数
    uint32_t total_bytes_input = plaintext_tensor->GetShape().GetShapeSize();

    if (total_bytes_input == 0) {
        // printf("[INFO] TilingFunc: Input plaintext size is 0.\n");
        // 即使输入为空，TilingData也应正确填充，Kernel侧会处理这种情况。
    }

    // 2. 获取属性值
    int32_t attr_use_core_num = 32;
    (void)context->GetAttr("useCoreNum", attr_use_core_num);

    int32_t attr_pipeline_tile_bytes_config = 1024; // 属性名从 Halfs 改为 Bytes
    (void)context->GetAttr("pipelineProcessTileBytes", attr_pipeline_tile_bytes_config);

    if (attr_pipeline_tile_bytes_config <= 0) {
        // printf("[ERROR] TilingFunc: Attribute pipelineProcessTileBytes (%d) must be positive.\n", attr_pipeline_tile_bytes_config);
        ERROR_LOG("Attribute pipelineProcessTileBytes (%d) must be positive.", attr_pipeline_tile_bytes_config);
        return ge::GRAPH_FAILED;
    }
    uint32_t actual_config_pipeline_tile_bytes = static_cast<uint32_t>(attr_pipeline_tile_bytes_config);

    // 可选：确保pipeline处理的tile大小至少是一个AES块，并且是对齐的
    if (actual_config_pipeline_tile_bytes < AES_BLOCK_BYTES_CONST && total_bytes_input > 0) { // 仅当有数据时调整
        // printf("[WARN] TilingFunc: pipelineProcessTileBytes (%u) is less than AES block size in bytes (%u). Adjusting to %u.\n",
        //        actual_config_pipeline_tile_bytes, AES_BLOCK_BYTES_CONST, AES_BLOCK_BYTES_CONST);
        actual_config_pipeline_tile_bytes = AES_BLOCK_BYTES_CONST;
    }
    if (actual_config_pipeline_tile_bytes % AES_BLOCK_BYTES_CONST != 0 && total_bytes_input > 0) { // 仅当有数据时调整
        // printf("[WARN] TilingFunc: pipelineProcessTileBytes (%u) is not a multiple of AES block size in bytes (%u). Adjusting upwards.\n",
        //        actual_config_pipeline_tile_bytes, AES_BLOCK_BYTES_CONST);
        actual_config_pipeline_tile_bytes = ((actual_config_pipeline_tile_bytes + AES_BLOCK_BYTES_CONST - 1) / AES_BLOCK_BYTES_CONST) * AES_BLOCK_BYTES_CONST;
    }
    // 如果total_bytes_input为0, actual_config_pipeline_tile_bytes可以保持用户设置或设为0/默认值，Kernel会处理
    if (total_bytes_input == 0) {
        actual_config_pipeline_tile_bytes = 0; // 或保持用户设置，Kernel会判断
    }


    // 3. 确定实际使用的核心数
    uint32_t framework_suggested_block_dim = context->GetBlockDim();
    uint32_t actual_block_dim_to_use = framework_suggested_block_dim;

    if (attr_use_core_num > 0 && static_cast<uint32_t>(attr_use_core_num) < framework_suggested_block_dim) {
        actual_block_dim_to_use = static_cast<uint32_t>(attr_use_core_num);
    }

    // 如果没有数据，理论上可以用0个或1个核，Kernel不执行。
    // 如果有数据但计算出的核数为0，则至少用1核。
    if (total_bytes_input > 0 && actual_block_dim_to_use == 0) {
        actual_block_dim_to_use = 1;
    } else if (total_bytes_input == 0) {
        actual_block_dim_to_use = (attr_use_core_num > 0 && static_cast<uint32_t>(attr_use_core_num) < framework_suggested_block_dim) ? static_cast<uint32_t>(attr_use_core_num) : framework_suggested_block_dim;
        if(actual_block_dim_to_use == 0) actual_block_dim_to_use = 1; // 至少为1，即使没数据
    }
    context->SetBlockDim(actual_block_dim_to_use);

    // 4. 填充TilingData
    AesCtrCustomTilingData tiling_data;

    // 填充全局总长度 (bytes)
    tiling_data.set_global_total_bytes(total_bytes_input);

    // 填充配置的（可能已调整对齐的）单个流水线tile的大小 (bytes)
    tiling_data.set_config_pipeline_tile_bytes(actual_config_pipeline_tile_bytes);

    // 5. 序列化TilingData并设置Workspace
    gert::RunTimeRawTilingData* raw_tiling_data = context->GetRawTilingData();
    if (raw_tiling_data == nullptr) {
        ERROR_LOG("Failed to get raw tiling data.");
        return ge::GRAPH_FAILED;
    }
    tiling_data.SaveToBuffer(raw_tiling_data->GetData(),
                             raw_tiling_data->GetCapacity());
    raw_tiling_data->SetDataSize(tiling_data.GetDataSize());

    context->SetWorkspaceSize(0);

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
// Shape推导函数
static graphStatus InferShapeAesCtrCustom(Operator& op) {
    TensorDesc p_desc = op.GetInputDesc("plaintext");
    Shape p_shape = p_desc.GetShape(); // 对于uint8类型，shape通常表示元素数量

    TensorDesc c_desc = op.GetOutputDesc("ciphertext");
    c_desc.SetShape(p_shape); // 输出形状与输入一致
    c_desc.SetOriginShape(p_desc.GetOriginShape());
    op.UpdateOutputDesc("ciphertext", c_desc);

    TensorDesc key_desc = op.GetInputDesc("key");
    if (key_desc.GetShape().GetDimNum() != 1 || key_desc.GetShape().GetDim(0) != 16) {
        // std::string shape_str = DebugString(key_desc.GetShape().GetDims());
        // printf("[ERROR] InferShape: Input 'key' shape %s is invalid, expected 1D with 16 elements.\n", shape_str.c_str());
        ERROR_LOG("Input 'key' shape is invalid, expected 1D with 16 elements (bytes).");
        return GRAPH_FAILED;
    }
    TensorDesc iv_desc = op.GetInputDesc("iv");
    if (iv_desc.GetShape().GetDimNum() != 1 || iv_desc.GetShape().GetDim(0) != 16) {
        // std::string shape_str = DebugString(iv_desc.GetShape().GetDims());
        // printf("[ERROR] InferShape: Input 'iv' shape %s is invalid, expected 1D with 16 elements.\n", shape_str.c_str());
        ERROR_LOG("Input 'iv' shape is invalid, expected 1D with 16 elements (bytes).");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// 数据类型推导函数
static graphStatus InferDataTypeAesCtrCustom(Operator& op) {
    DataType plaintext_dtype = op.GetInputDesc("plaintext").GetDataType();
    DataType key_dtype = op.GetInputDesc("key").GetDataType();
    DataType iv_dtype = op.GetInputDesc("iv").GetDataType();

    // 校验输入数据类型
    if (plaintext_dtype != DT_UINT8) { // 明文应为 DT_UINT8
        ERROR_LOG("Input 'plaintext' data type must be DT_UINT8.");
        return GRAPH_FAILED;
    }
    if (key_dtype != DT_UINT8) { // 密钥应为 DT_UINT8
        ERROR_LOG("Input 'key' data type must be DT_UINT8.");
        return GRAPH_FAILED;
    }
    if (iv_dtype != DT_UINT8) { // IV应为 DT_UINT8
        ERROR_LOG("Input 'iv' data type must be DT_UINT8.");
        return GRAPH_FAILED;
    }

    // 更新输出数据类型
    // 密文数据类型与明文一致
    op.UpdateOutputDesc("ciphertext", op.GetInputDesc("plaintext").GetDataType());
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AesCtrCustom : public OpDef {
public:
    AesCtrCustom(const char* name) : OpDef(name) {
        this->Input("plaintext")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8}) // 明文数据类型改为 DT_UINT8
            .Format({ge::FORMAT_ND}); // 通常为1D byte array
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})   // 密钥数据类型为 DT_UINT8
            .Format({ge::FORMAT_ND});   // 通常为1D [16] byte array
        this->Input("iv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})    // IV数据类型为 DT_UINT8
            .Format({ge::FORMAT_ND});    // 通常为1D [16] byte array
        this->Output("ciphertext")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8}) // 密文数据类型改为 DT_UINT8
            .Format({ge::FORMAT_ND});

        this->Attr("useCoreNum")
            .ParamType(OPTIONAL)
            .DataType(INT)
            .DefaultValue(32);
        this->Attr("pipelineProcessTileBytes") // 属性名从 Halfs 改为 Bytes
            .ParamType(OPTIONAL)
            .DataType(INT)
            .DefaultValue(1024); // 默认tile大小，单位：byte

        this->SetInferShapeFn(ge::InferShapeAesCtrCustom);
        this->SetInferDataTypeFn(ge::InferDataTypeAesCtrCustom);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910");
                                    
};
OP_ADD(AesCtrCustom);
} // namespace ops
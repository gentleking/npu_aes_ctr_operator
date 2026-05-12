/**
 * @file aes_ctr_decrypt.cpp
 * @brief Host 侧算子实现 —— 原型注册、Shape/DataType 推导、Tiling
 *
 * 本文件运行在 CPU（Host）侧，负责三件事：
 *
 * 1. 算子原型定义 (OpDef)
 *    告诉 CANN 框架：本算子叫什么名字、有几个输入/输出、
 *    各 tensor 的数据类型和格式是什么。框架根据这个定义做
 *    图编译和资源分配。
 *
 * 2. Shape/DataType 推导 (InferShape / InferDataType)
 *    当框架编译计算图时，需要知道算子的输出形状和数据类型。
 *    这里直接将输出的 shape 和 dtype 设为与输入一致。
 *
 * 3. Tiling (TilingFunc)
 *    在 kernel 执行前，Host 侧先计算 tile 参数（如数据分块大小），
 *    通过 tiling buffer 传给 NPU 侧的 kernel。NPU 的 Local Memory
 *    有限，无法一次处理全部数据，需要分块流水线处理。
 *
 * NPU 算子开发与 CPU 开发的关键区别：
 *   - CPU 开发：一层代码完成所有逻辑，数据在 RAM 中，随意访问
 *   - NPU 开发：分为 Host 侧（本文件）和 Device 侧（op_kernel/）
 *     Host 侧负责原型注册和 tiling，Device 侧负责实际计算
 *   - Host 和 Device 通过 tiling buffer 通信，传输计算参数
 *   - NPU Kernel 受 Local Memory 大小限制，必须分 tile 处理
 */

#include "aes_ctr_decrypt_tiling.h"
#include "register/op_def_registry.h"


/**
 * @namespace optiling
 * @brief Tiling 函数所在的命名空间
 *
 * TilingFunc 在算子执行前由 Host 调用，填充 TilingData 结构体，
 * 序列化后传给 kernel。
 */
namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  // 1. 创建 tiling 数据实例
  AesCtrDecryptTilingData tiling;

  // 2. 获取输入张量的元素总数
  //    GetShapeSize() 返回 uint8 元素个数，即字节数
  //    例如 shape = {8, 2048} 的 uint8 tensor → 16384
  //    这个值会通过 tiling buffer 传给 kernel，
  //    kernel 的 Init() 用 totalLength/2 得到 half 单元数
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

  // 3. 设置 block 维度（并行计算的核数）
  //    Ascend 310P3 有 8 个 AI Core，这里设为 8 核并行
  //    kernel 侧通过 AscendC::GetBlockNum() 获取此值
  context->SetBlockDim(8);

  // 4. 填充 tiling 字段
  tiling.set_totalLength(totalLength);
  tiling.set_tileNum(8);

  // 5. 将 tiling 数据序列化到 buffer 中，传给 kernel
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // 6. 设置 workspace 大小（本算子不需要额外 workspace）
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

  return ge::GRAPH_SUCCESS;
}

}


/**
 * @namespace ge
 * @brief Shape 和 DataType 推导函数
 *
 * InferShape: 输出形状 = 输入形状（解密不改变数据形状）
 * InferDataType: 输出类型 = 输入类型（uint8 → uint8）
 */
namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
  // 取第 0 个输入（cyphertext）的形状，直接赋给第 0 个输出（plaintext）
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y_shape = context->GetOutputShape(0);
  *y_shape = *x1_shape;
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
  // 取第 0 个输入的数据类型，直接赋给第 0 个输出
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}

}


/**
 * @namespace ops
 * @brief 算子原型定义
 *
 * OpDef 是 Ascend 算子注册的核心类，描述算子的接口信息：
 * - 输入/输出 tensor 的名称、数据类型、格式
 * - 支持的硬件平台（AICore 类型）
 * - 关联的 shape/data type 推导函数和 tiling 函数
 *
 * OP_ADD 宏将算子注册到全局算子表中，
 * 框架在编译计算图时通过名字查找并调用此算子。
 */
namespace ops {

class AesCtrDecrypt : public OpDef {
public:
  explicit AesCtrDecrypt(const char* name) : OpDef(name)
  {
    // ---- 输入定义 ----
    // 输入 0: cyphertext, 待解密的密文
    //   - ParamType(REQUIRED): 必选输入
    //   - DataType({ge::DT_UINT8}): 数据类型为 uint8
    //   - Format({ge::FORMAT_ND}): 数据格式为 ND（N-Dimension，即任意维度）
    //   - UnknownShapeFormat: 动态形状时使用的格式
    this->Input("cyphertext")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    // 输入 1: key, AES-128 密钥
    //   固定为 16 字节 uint8
    this->Input("key")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    // ---- 输出定义 ----
    // 输出 0: plaintext, 解密后的明文
    //   形状和数据类型与输入一致
    this->Output("plaintext")
        .ParamType(REQUIRED)
        .DataType({ge::DT_UINT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    // ---- 注册 shape/data type 推导函数 ----
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    // ---- 配置 AI Core 执行 ----
    // .AICore() 表示此算子在 AI Core 上执行（而非 AI CPU）
    // SetTiling 注册 tiling 函数
    // AddConfig 声明支持的处理器型号
    this->AICore()
        .SetTiling(optiling::TilingFunc);

    // 支持两种 NPU 芯片：
    //   - ascend910: 训练卡，8 核 AI Core
    //   - ascend310p: 推理卡，8 核 AI Core（本测试环境使用的硬件）
    this->AICore().AddConfig("ascend910")
                  .AddConfig("ascend310p");
  }
};

// OP_ADD(类名) 宏将算子注册到 CANN 框架中
// 框架通过算子名字 "AesCtrDecrypt" 查找和调用
// 名字需要与 AesCtrDecrypt.json 中的 "op" 字段一致
OP_ADD(AesCtrDecrypt);

}

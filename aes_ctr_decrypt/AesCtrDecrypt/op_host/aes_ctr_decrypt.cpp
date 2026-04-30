
#include "aes_ctr_decrypt_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AesCtrDecryptTilingData tiling;
  uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
//   const gert::StorageShape* x1_shape = context->GetInputShape(0);
//   int32_t data_sz = 1;
//   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
//     data_sz *= x1_shape->GetStorageShape().GetDim(i);
//   tiling.set_size(data_sz);
    context->SetBlockDim(8);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(8);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AesCtrDecrypt : public OpDef {
public:
    explicit AesCtrDecrypt(const char* name) : OpDef(name)
    {
        this->Input("cyphertext")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("plaintext")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910")
                      .AddConfig("ascend310p");

    }
};

OP_ADD(AesCtrDecrypt);
}

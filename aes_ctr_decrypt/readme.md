# AES-CTR Decrypt
使用AES-CTR算子对一个Tensor进行解密

## Aes-Ctr算法
参考链接
算子分析
- 输入：明文plaintext，密钥key，计数值iv
- 输出：密文cyphertext
算子核函数实现
需要考虑的因素：
**Tiling**
**Data Size**
**Local Memory**
**实现类KernelAesCtr**
1. 构造函数KernelAesCtr()
- 初始化 sbox 数组，设为固定常量
- 将常量数组拷贝到私有成员 sbox
2. 初始化函数Init(GM_ADDR plaintext, GM_ADDR key, GM_ADDR iv, GM_ADDR ciphertext)
初始化函数中，需要设置GM全局缓冲区，初始化双缓冲队列，并从GM复制密钥和IV，执行密钥扩展。
- plaintext    GM中存储明文的地址
- key           GM中存储AES-128密钥的地址
- iv              GM中存储初始向量(IV)的地址
- ciphertext  GM中存储加密结果的地址
- 通过 GlobalTensor 设置 plaintextGm、ciphertextGm 的起始地址和长度。
- 调用 TPipe::InitBuffer() 初始化多缓冲队列，以实现流水线并行。
- 从GM复制key、iv到本地，然后调用KeyExpansion进行密钥扩展。
- 根据blockIdx对IV做偏移处理，以便多核并行时能使用不同的计数器起点。
3. 处理函数Process()
依次执行数据读入、加密计算、数据写出。其中包含TILE_NUM * BUFFER_NUM次循环，每次循环都完成一次CopyIn、Compute、CopyOut的排布。在硬件层面，CopyIn、Compute、CopyOut三个阶段可以流水线并行执行，从而提升吞吐量。
- CopyIn()
  - 函数功能：数据读入。从GM复制数据到本地Tensor
  - 流水线Stage 1：本函数对应“CopyIn”阶段，将全局内存（GM）中的数据搬运到Local Memory。使用 TQue 的 AllocTensor<half>() 分配一个可存储 TILE_LENGTH 大小的本地Tensor。使用 DataCopy() 将 GM 中的明文数据复制到本地Tensor。将装载了数据的本地Tensor 放入（EnQue）队列 inQueuePlain，供下一个Stage（Compute）使用。
- Compute()
  - 函数功能：加密计算。从本地Tensor读取数据，执行AES-CTR加密，再写回加密结果到本地Tensor
  - 流水线Stage 2：本函数对应“Compute”阶段，读取队列 inQueuePlain 中的本地Tensor，进行AES-CTR加密，然后将结果写入另一个本地Tensor，再放入 outQueueCipher 队列，等待下一阶段输出。DeQue() 从 inQueuePlain 中获取上一阶段传入的明文Tensor。分配新的本地Tensor（cipherLocal）以存放加密结果。进行必要的数据格式转换（half -> uint8_t 数组），再调用 AesCtrEncrypt。将加密结果转换回 half 并写入 cipherLocal。EnQue() 将加密后的数据放入 outQueueCipher。FreeTensor() 释放用完的明文Tensor 以复用本地内存。
- CopyOut()
  - 函数功能：数据写出。将加密结果从本地Tensor复制回GM
  - 流水线Stage 3：本函数对应“CopyOut”阶段，从 outQueueCipher 中取出已经完成加密的本地Tensor，并使用 DataCopy() 写回至 GM 中的目标地址，最后释放本地Tensor。DeQue() 从 outQueueCipher 获取已加密的本地Tensor。DataCopy() 将加密结果写回到 GM（ciphertextGm）。FreeTensor() 释放本地Tensor 以复用空间。
## 直调模式调用算子
### 算子直调性能测试
算子加解密不同数据量运行时间图
## 算子工程开发
### 新建算子工程
参考Ascend C 算子开发
在Ascend-toolkit中提供了自定义算子工程生成工具msOpGen，可基于算子原型定义输出算子工程：包括算子host侧代码实现文件、算子kernel侧实现文件以及工程编译配置文件等。具体创建步骤如下。
1. 编写算子的原型定义json文件，用于生成算子开发工程。以AddCustom算子为例，编写json文件并命名为add_custom.json，文件内容如下：
```json
[
    {
        "op": "AddCustom",
        "input_desc": [
            {
                "name": "x",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            },
            {
                "name": "y",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            }
        ],
        "output_desc": [
            {
                "name": "z",
                "param_type": "required",
                "format": [
                    "ND",
                    "ND",
                    "ND"
                ],
                "type": [
                    "float16",
                    "float",
                    "int32"
                ]
            }
        ]
    }
]
```
配置文件具体参数含义可参考 链接。
2. 根据配置文件，使用msOpGen工具生成算子的开发工程。以生成AddCustom的算子工程为例，具体命令为：
```bash
/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen gen -i $HOME/sample/add_custom.json -c ai_core-Ascend910A -lan cpp -out $HOME/sample/AddCustom
```
具体参数解释：
- -i：指定算子原型定义文件add_custom.json所在路径，请根据实际情况修改。
- -c：ai_core-Ascend910A代表算子在AI Core上执行，Ascend910A为昇腾AI处理器的型号。
- -lan： 参数cpp代表算子基于Ascend C编程框架，使用C/C++编程语言开发。
- -out：生成文件所在路径，可配置为绝对路径或者相对路径，并且工具执行用户对路径具有可读写权限。若不配置，则默认生成在执行命令的当前路径。
3. 最终生成的算子工程中包含算子实现的模板文件，编译脚本等，以AddCustom算子为例，目录结构如下所示：
```
AddCustom
├── build.sh         // 编译入口脚本
├── cmake            // 算子工程编译所需脚本及公共编译文件存放目录
├── CMakeLists.txt    // 算子工程的CMakeLists.txt
├── CMakePresets.json    // 编译配置项
├── framework        // AI框架适配时，算子插件实现文件目录
├── op_host                      // host侧实现文件
│   ├── add_custom_tiling.h    // 算子tiling定义文件
│   ├── add_custom.cpp         // 算子原型注册、shape推导、信息库、tiling实现等内容文件
│   ├── CMakeLists.txt
├── op_kernel                   // kernel侧实现文件
│   ├── CMakeLists.txt   
│   ├── add_custom.cpp        // 算子代码实现文件 
└── scripts                     // 自定义算子工程打包相关脚本所在目录
```
注意：
- 上述目录结构中的粗体下划线文件为后续算子开发过程中需要修改的文件，其他文件无需修改。
- kernel侧实现依赖的所有文件需全部放置在op_kernel目录下，不遵循该约束会导致源码发布场景在线编译失败。因为在后续的算子源码发布时，只会打包发布op_kernel目录下的文件。
- 工程目录中的op_kernel和op_host包含了算子的核心实现文件。op_kernel存放kernel侧算子实现。op_host存放host侧代码实现，包括算子原型定义、host侧tiling实现等。
### 算子的实现、编译和安装
参考Ascend C 算子开发
### ACLNN模式调用算子
参考Ascend C 算子开发
### Pytorch模式调用算子
PyTorch官方提供的native_functions.yaml文件定义了PyTorch Native Functions的具体算子定义和分发细节，定义则通过.cpp文件实现。OpPlugin仓库与原生类似，使用yaml文件定义了NPU适配的算子，算子具体适配则存放在.cpp文件中。
适配算子主要分为两步，以torch API abs/abs_out为例，包含基于aclnn算子和aclop算子，适配包括两部分，一是算子接口yaml配置，二是算子kernel的适配代码：
1) 在yaml文件中配置算子。（op_plugin_functions.yaml）
OpPlugin采用和原生PyTorch类似的逻辑在yaml中声明算子的各类信息，通过在yaml中配置算子，自动生成算子声明和注册代码。算子的Aten IR定义位于op_plugin/config/op_plugin_functions.yaml文件中，所有版本的定义都在这个文件里面，通过配置不同版本来区分。
2) 算子适配实现。（AbsKernelNpuOpApi.cpp）
当前支持适配基于aclnn算子和aclop算子两类算子，aclnn算子是较新算子实现方式，推荐使用该方式，其适配文件位于op_plugin/ops/opapi目录；aclop算子是早期的算子实现方式，不推荐使用，其适配文件位于op_plugin/ops/aclops目录。
一个算子所有版本的适配代码都在一个文件中，通过编译宏VERSION_BETWEEN来区分不同版本。
新增自定义算子需要同步新增算子适配文件，并参考如下示例进行相关算子实现的开发。
aclnn算子适配：如果所有版本的适配代码一致，则不需要额外添加编译宏，适配文件路径为：op_plugin/ops/opapi/AbsKernelNpuOpApi.cpp，文件命名规范为算子名称+KernelNpuOpApi，算子名称首字母大写。
3) 运行脚本```run_op_plugin.sh```
### 算子功能测试和性能测试
- 采集和分析运行在昇腾AI处理器上算子的关键性能指标工具参考链接
- 在真实的硬件环境中，对算子的输入输出进行测试，以验证算子的功能是否正确。可选择使用msOpST工具进行ST（System Test）测试，参考链接
# AesCtrDecrypt算子开发
同AesCtrEncrypt算子开发
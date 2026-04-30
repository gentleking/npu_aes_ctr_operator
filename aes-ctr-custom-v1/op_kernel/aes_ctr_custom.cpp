/**
 * @file aes_ctr_custom.cpp
 * @brief Ascend-C - AES-CTR 自定义算子（尾块集中最后一核处理 • 最终版）
 *
 * 关键特性
 * ──────────────────────────────────────────────────────────────
 * • 仅前 N-1 核保证 32 B（16 half）对齐；全局尾差交给最后一核。
 *   ⇒ 总长度只要 16 B 对齐即可。
 * • 使用 LocalTensor::GetValue / SetValue 正确访问 half 数据，
 *   避免 “address-of-temporary” 编译错误。
 */

#include "kernel_operator.h"
#include <cstring>

/* ────────── 可调配置 ────────── */
constexpr int32_t TOTAL_LENGTH = 1040;   // 全局 half 数（✱16 B 对齐即可）
constexpr int32_t USE_CORE_NUM = 2;      // AICore 个数

constexpr int32_t TILE_NUM     = 2;      // 每缓冲区 tile 数
constexpr int32_t BUFFER_NUM   = 2;      // 双缓冲

constexpr int32_t ALIGN_HALF   = 16;     // 16 half = 32 B 传输粒度

/* 前 N-1 核：向下取整到 32 B；最后一核吃掉余量 */
constexpr int32_t FIXED_HALF =
    (TOTAL_LENGTH / USE_CORE_NUM / ALIGN_HALF) * ALIGN_HALF;
constexpr int32_t LAST_CORE_HALF =
    TOTAL_LENGTH - FIXED_HALF * (USE_CORE_NUM - 1);

/* tile 大小上界，用于静态缓冲区 */
constexpr int32_t MAX_TILE_HALF  = FIXED_HALF / (TILE_NUM * BUFFER_NUM);
constexpr int32_t MAX_TILE_BYTES = MAX_TILE_HALF * 2;

/* ────────── AES 常量 ────────── */
static const int AES_KEY_BYTES          = 16;
static const int AES_BLOCK_BYTES        = 16;
static const int AES_ROUND_NUM          = 10;
static const int AES_EXPANDED_KEY_BYTES = 16 * (AES_ROUND_NUM + 1);

/* ────────── AES 工具函数（与原版一致，略去注释） ────────── */
__aicore__ inline uint8_t xtime(uint8_t x)
{ return (uint8_t)((x << 1) ^ ((x & 0x80) ? 0x1B : 0x00)); }

static __aicore__ void KeyExpansion(const uint8_t *key,
                                    uint8_t *roundKeys,
                                    const uint8_t *sbox)
{
    uint8_t Rcon[11] = {0x00,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36};
    for (int i = 0; i < AES_KEY_BYTES; ++i) roundKeys[i] = key[i];

    int used = AES_KEY_BYTES, rIdx = 1;  uint8_t tmp[4];
    while (used < AES_EXPANDED_KEY_BYTES) {
        for (int i = 0; i < 4; ++i) tmp[i] = roundKeys[used - 4 + i];
        if ((used % AES_KEY_BYTES) == 0) {
            uint8_t t = tmp[0];
            tmp[0] = (uint8_t)(sbox[tmp[1]] ^ Rcon[rIdx]);
            tmp[1] = sbox[tmp[2]];
            tmp[2] = sbox[tmp[3]];
            tmp[3] = sbox[t];
            ++rIdx;
        }
        for (int i = 0; i < 4; ++i) {
            roundKeys[used] = (uint8_t)(roundKeys[used - AES_KEY_BYTES] ^ tmp[i]);
            ++used;
        }
    }
}

static __aicore__ inline void AddRoundKey(uint8_t *s, const uint8_t *rk)
{ for (int i = 0; i < 16; ++i) s[i] ^= rk[i]; }

static __aicore__ void SubShiftMix(uint8_t *s, uint32_t, const uint8_t *sb)
{
    for (int i = 0; i < 16; ++i) s[i] = sb[s[i]];
    uint8_t t,tmp2;
    t=s[1];s[1]=s[5];s[5]=s[9];s[9]=s[13];s[13]=t;
    t=s[2];tmp2=s[6];s[2]=s[10];s[6]=s[14];s[10]=t;s[14]=tmp2;
    t=s[3];s[3]=s[15];s[15]=s[11];s[11]=s[7];s[7]=t;
    for(int col=0;col<4;++col){
        int idx=col*4;
        uint8_t a0=s[idx],a1=s[idx+1],a2=s[idx+2],a3=s[idx+3];
        uint8_t T=a0^a1^a2^a3,U=a0;
        s[idx]^= T ^ xtime(a0^a1);
        s[idx+1]^= T ^ xtime(a1^a2);
        s[idx+2]^= T ^ xtime(a2^a3);
        s[idx+3]^= T ^ xtime(a3^U);
    }
}

static __aicore__ void SubShift(uint8_t *s,const uint8_t *sb)
{
    for(int i=0;i<16;++i) s[i]=sb[s[i]];
    uint8_t t,tmp2;
    t=s[1];s[1]=s[5];s[5]=s[9];s[9]=s[13];s[13]=t;
    t=s[2];tmp2=s[6];s[2]=s[10];s[6]=s[14];s[10]=t;s[14]=tmp2;
    t=s[3];s[3]=s[15];s[15]=s[11];s[11]=s[7];s[7]=t;
}

static __aicore__ void AesEncryptBlock(const uint8_t in[16],
                                       const uint8_t rk[AES_EXPANDED_KEY_BYTES],
                                       uint8_t out[16], uint32_t blkId,
                                       const uint8_t *sbox)
{
    uint8_t st[16];
    for(int i=0;i<16;++i) st[i]=in[i];
    AddRoundKey(st,rk);
    for(int r=1;r<AES_ROUND_NUM;++r){ SubShiftMix(st,blkId,sbox); AddRoundKey(st,rk+16*r); }
    SubShift(st,sbox); AddRoundKey(st,rk+16*AES_ROUND_NUM);
    for(int i=0;i<16;++i) out[i]=st[i];
}

__aicore__ inline void BigEndAdd(uint8_t *ctr,uint32_t val)
{
    int carry=0;
    for(int i=15;i>=0;--i){
        uint32_t sum=(uint32_t)ctr[i]+(val&0xFF)+carry;
        ctr[i]=(uint8_t)(sum&0xFF);
        carry=(sum>>8)&0xFF;
        val>>=8;
        if(!val&&!carry) break;
    }
}

static __aicore__ void AesCtrEncrypt(const uint8_t *pt,uint8_t *ct,int32_t lenB,
                                     const uint8_t *rk,uint8_t iv[16],int32_t blkId,
                                     const uint8_t *sbox)
{
    uint8_t ctr[16]; for(int i=0;i<16;++i) ctr[i]=iv[i];
    uint8_t ks[16]; int off=0;
    while(off<lenB){
        AesEncryptBlock(ctr,rk,ks,blkId,sbox);
        int bs=(lenB-off<16)?(lenB-off):16;
        for(int i=0;i<bs;++i) ct[off+i]=pt[off+i]^ks[i];
        off+=bs;
        for(int i=15;i>=0;--i){ ctr[i]++; if(ctr[i]) break; }
    }
    for(int i=0;i<16;++i) iv[i]=ctr[i];
}

/* ────────── NPU 路径 ────────── */
#ifndef ASCENDC_CPU_DEBUG
class KernelAesCtr {
public:
    __aicore__ inline KernelAesCtr();
    __aicore__ inline void Init(GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int idx);
    __aicore__ inline void Compute(int idx);
    __aicore__ inline void CopyOut(int idx);
    __aicore__ inline void EncryptTail();

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN ,BUFFER_NUM> inQ;
    AscendC::TQue<AscendC::QuePosition::VECOUT,BUFFER_NUM> outQ;
    AscendC::GlobalTensor<half> plainGm,cipherGm;

    uint8_t IV [AES_BLOCK_BYTES];
    uint8_t rk [AES_EXPANDED_KEY_BYTES];
    uint8_t sbox[256];

    int32_t tileLen=0,processedHalf=0,tailHalf=0;
};

__aicore__ inline KernelAesCtr::KernelAesCtr()
{
    uint8_t t[256]={
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,
        0xfe,0xd7,0xab,0x76,0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,
        0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,0xb7,0xfd,0x93,0x26,
        0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,
        0xeb,0x27,0xb2,0x75,0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,
        0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,0x53,0xd1,0x00,0xed,
        0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,
        0x50,0x3c,0x9f,0xa8,0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,
        0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,0xcd,0x0c,0x13,0xec,
        0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,
        0xde,0x5e,0x0b,0xdb,0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,
        0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,0xe7,0xc8,0x37,0x6d,
        0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,
        0x4b,0xbd,0x8b,0x8a,0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,
        0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,0xe1,0xf8,0x98,0x11,
        0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,
        0xb0,0x54,0xbb,0x16};
    for(int i=0;i<256;++i) sbox[i]=t[i];
}

__aicore__ inline void KernelAesCtr::Init(GM_ADDR p,GM_ADDR k,GM_ADDR v,GM_ADDR c)
{
    int blk = AscendC::GetBlockIdx();

    int32_t myHalf     = (blk == USE_CORE_NUM-1)? LAST_CORE_HALF : FIXED_HALF;
    int32_t offsetHalf = FIXED_HALF * blk;

    plainGm .SetGlobalBuffer((__gm__ half*)p + offsetHalf, myHalf);
    cipherGm.SetGlobalBuffer((__gm__ half*)c + offsetHalf, myHalf);

    tileLen = myHalf / (TILE_NUM * BUFFER_NUM);
    tileLen -= tileLen % 8;
    if(tileLen < 8) tileLen = 8;

    processedHalf = tileLen * TILE_NUM * BUFFER_NUM;
    tailHalf      = myHalf - processedHalf;

    pipe.InitBuffer(inQ ,BUFFER_NUM,tileLen*sizeof(half));
    pipe.InitBuffer(outQ,BUFFER_NUM,tileLen*sizeof(half));

    uint8_t localKey[AES_KEY_BYTES];
    for(int i=0;i<AES_KEY_BYTES;++i) localKey[i]=reinterpret_cast<__gm__ uint8_t*>(k)[i];
    for(int i=0;i<16;++i) IV[i]=reinterpret_cast<__gm__ uint8_t*>(v)[i];
    KeyExpansion(localKey,rk,sbox);

    uint32_t offBlk=(FIXED_HALF*2)/AES_BLOCK_BYTES;
    uint8_t tmp[16]; for(int i=0;i<16;++i) tmp[i]=IV[i];
    BigEndAdd(tmp, offBlk*blk);
    for(int i=0;i<16;++i) IV[i]=tmp[i];
}

__aicore__ inline void KernelAesCtr::CopyIn(int idx)
{
    AscendC::LocalTensor<half> t = inQ.AllocTensor<half>();
    AscendC::DataCopy(t, plainGm[idx*tileLen], tileLen);
    inQ.EnQue(t);
}

__aicore__ inline void KernelAesCtr::Compute(int /*idx*/)
{
    int blk = AscendC::GetBlockIdx();
    auto plain  = inQ.DeQue<half>();
    auto cipher = outQ.AllocTensor<half>();

    uint8_t plainBuf[MAX_TILE_BYTES];
    for(int i=0;i<tileLen;++i){
        half hv = plain.GetValue(i);
        uint16_t w = *reinterpret_cast<uint16_t*>(&hv);
        plainBuf[2*i]   = (uint8_t) w;
        plainBuf[2*i+1] = (uint8_t)(w >> 8);
    }

    uint8_t cipherBuf[MAX_TILE_BYTES];
    AesCtrEncrypt(plainBuf,cipherBuf,tileLen*2,rk,IV,blk,sbox);

    for(int i=0;i<tileLen;++i){
        uint16_t w = (uint16_t)cipherBuf[2*i] |
                     ((uint16_t)cipherBuf[2*i+1] << 8);
        cipher.SetValue(i,*reinterpret_cast<half*>(&w));
    }
    outQ.EnQue(cipher);
    inQ.FreeTensor(plain);
}

__aicore__ inline void KernelAesCtr::CopyOut(int idx)
{
    auto t = outQ.DeQue<half>();
    AscendC::DataCopy(cipherGm[idx*tileLen], t, tileLen);
    outQ.FreeTensor(t);
}

__aicore__ inline void KernelAesCtr::EncryptTail()
{
    if(tailHalf <= 0) return;

    auto plain  = inQ.AllocTensor<half>();
    auto cipher = outQ.AllocTensor<half>();
    AscendC::DataCopy(plain, plainGm[processedHalf], tailHalf);

    uint8_t pt [MAX_TILE_BYTES] = {0};
    for(int i=0;i<tailHalf;++i){
        half hv = plain.GetValue(i);
        uint16_t w = *reinterpret_cast<uint16_t*>(&hv);
        pt[2*i]   = (uint8_t) w;
        pt[2*i+1] = (uint8_t)(w >> 8);
    }
    uint8_t ct[MAX_TILE_BYTES];
    AesCtrEncrypt(pt,ct,tailHalf*2,rk,IV,AscendC::GetBlockIdx(),sbox);

    for(int i=0;i<tailHalf;++i){
        uint16_t w = (uint16_t)ct[2*i] |
                     ((uint16_t)ct[2*i+1] << 8);
        cipher.SetValue(i,*reinterpret_cast<half*>(&w));
    }
    AscendC::DataCopy(cipherGm[processedHalf], cipher, tailHalf);
    outQ.FreeTensor(cipher);
    inQ.FreeTensor(plain);
}

__aicore__ inline void KernelAesCtr::Process()
{
    for(int i=0;i<TILE_NUM*BUFFER_NUM;++i){ CopyIn(i); Compute(i); CopyOut(i); }
    EncryptTail();
}

extern "C" __global__ __aicore__
void aes_ctr_custom(GM_ADDR p,GM_ADDR k,GM_ADDR v,GM_ADDR c)
{
    KernelAesCtr op; op.Init(p,k,v,c); op.Process();
}

/* —— Host launcher —— */
void aes_ctr_custom_do(uint32_t blockDim, void *stream,
                       uint8_t *pt,uint8_t *key,uint8_t *iv,uint8_t *ct)
{
    aes_ctr_custom<<<blockDim,nullptr,stream>>>(pt,key,iv,ct);
}

#else  /* ────────── CPU-Debug 路径（与原版一致，保持不变） ────────── */
extern "C" __global__ __aicore__
void aes_ctr_custom(GM_ADDR plaintext,GM_ADDR key,GM_ADDR iv,GM_ADDR ciphertext)
{
    int32_t blk = AscendC::GetBlockIdx();
    int32_t myHalf = (blk == USE_CORE_NUM-1)? LAST_CORE_HALF : FIXED_HALF;
    int32_t offsetHalf = FIXED_HALF * blk;
    int32_t offsetBytes= offsetHalf * 2;

    uint8_t *ptAll = reinterpret_cast<uint8_t*>(plaintext);
    uint8_t *ctAll = reinterpret_cast<uint8_t*>(ciphertext);
    uint8_t *keyPtr= reinterpret_cast<uint8_t*>(key);
    uint8_t *ivPtr = reinterpret_cast<uint8_t*>(iv);

    uint8_t localKey[AES_KEY_BYTES];
    uint8_t localIV [AES_BLOCK_BYTES];
    std::memcpy(localKey,keyPtr,AES_KEY_BYTES);
    std::memcpy(localIV ,ivPtr ,AES_BLOCK_BYTES);

    uint32_t offBlk = (FIXED_HALF*2)/AES_BLOCK_BYTES;
    BigEndAdd(localIV, offBlk*blk);

    uint8_t sbox[256];
    for(int i=0;i<256;++i) sbox[i]=i;
    KeyExpansion(localKey, sbox, sbox);          // sbox 作两用
    uint8_t rk[AES_EXPANDED_KEY_BYTES];
    KeyExpansion(localKey,rk,sbox);

    int32_t tileLen = myHalf / (TILE_NUM*BUFFER_NUM);
    tileLen -= tileLen % 8;
    if(tileLen<8) tileLen=8;
    int32_t processedHalf=tileLen*TILE_NUM*BUFFER_NUM;
    int32_t tailHalf=myHalf-processedHalf;

    int32_t loop=TILE_NUM*BUFFER_NUM, tileBytes=tileLen*2;
    for(int i=0;i<loop;++i){
        const uint8_t* in=ptAll+offsetBytes+i*tileBytes;
        uint8_t*       o =ctAll+offsetBytes+i*tileBytes;
        AesCtrEncrypt(in,o,tileBytes,rk,localIV,blk,sbox);
    }
    if(tailHalf>0){
        const uint8_t* in=ptAll+offsetBytes+processedHalf*2;
        uint8_t*       o =ctAll+offsetBytes+processedHalf*2;
        AesCtrEncrypt(in,o,tailHalf*2,rk,localIV,blk,sbox);
    }
}
#endif /* ASCENDC_CPU_DEBUG */
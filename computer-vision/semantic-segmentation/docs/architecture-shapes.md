# Architecture Shapes

These diagrams assume the Oxford Pet config:

```text
input image: 3 x 256 x 256
num_classes: 2
```

For VOC, only the final classifier changes from `2` classes to `21` classes.

## U-Net

U-Net is an encoder-decoder CNN. The encoder reduces spatial size and increases
channels. The decoder upsamples and concatenates encoder features through skip
connections.

```text
Input
  image                         3 x 256 x 256
    |
    v
Encoder
  enc1 DoubleConv              32 x 256 x 256
    |
    | maxpool 2x2
    v
  enc2 DoubleConv              64 x 128 x 128
    |
    | maxpool 2x2
    v
  enc3 DoubleConv             128 x  64 x  64
    |
    | maxpool 2x2
    v
Bottleneck
  bottleneck DoubleConv        256 x  32 x  32
    |
    v
Decoder
  up3 ConvTranspose2d          128 x  64 x  64
  concat with enc3             256 x  64 x  64
  dec3 DoubleConv              128 x  64 x  64
    |
    v
  up2 ConvTranspose2d           64 x 128 x 128
  concat with enc2             128 x 128 x 128
  dec2 DoubleConv               64 x 128 x 128
    |
    v
  up1 ConvTranspose2d           32 x 256 x 256
  concat with enc1              64 x 256 x 256
  dec1 DoubleConv               32 x 256 x 256
    |
    v
Classifier
  1x1 convolution                2 x 256 x 256
```

### U-Net Shape Table

| Stage | Operation | Output shape |
|---|---|---|
| Input | image | `3 x 256 x 256` |
| Encoder 1 | DoubleConv | `32 x 256 x 256` |
| Down 1 | MaxPool | `32 x 128 x 128` |
| Encoder 2 | DoubleConv | `64 x 128 x 128` |
| Down 2 | MaxPool | `64 x 64 x 64` |
| Encoder 3 | DoubleConv | `128 x 64 x 64` |
| Down 3 | MaxPool | `128 x 32 x 32` |
| Bottleneck | DoubleConv | `256 x 32 x 32` |
| Up 3 | Transposed conv | `128 x 64 x 64` |
| Skip 3 | concat with enc3 | `256 x 64 x 64` |
| Decoder 3 | DoubleConv | `128 x 64 x 64` |
| Up 2 | Transposed conv | `64 x 128 x 128` |
| Skip 2 | concat with enc2 | `128 x 128 x 128` |
| Decoder 2 | DoubleConv | `64 x 128 x 128` |
| Up 1 | Transposed conv | `32 x 256 x 256` |
| Skip 1 | concat with enc1 | `64 x 256 x 256` |
| Decoder 1 | DoubleConv | `32 x 256 x 256` |
| Classifier | 1x1 conv | `2 x 256 x 256` |

## SegFormer

SegFormer is a transformer-based segmentation model. It does not keep a full
resolution feature map through the encoder. Instead, it creates hierarchical
transformer features at multiple scales, then a lightweight decoder fuses them.

For `nvidia/segformer-b0-finetuned-ade-512-512`, the encoder feature scales are:

```text
Input
  image                         3 x 256 x 256
    |
    v
SegFormer Encoder
  stage 1                      32 x  64 x  64   stride 4
  stage 2                      64 x  32 x  32   stride 8
  stage 3                     160 x  16 x  16   stride 16
  stage 4                     256 x   8 x   8   stride 32
    |
    v
SegFormer Decode Head
  project stage 1             256 x  64 x  64
  project stage 2             256 x  32 x  32 -> upsample to 64 x 64
  project stage 3             256 x  16 x  16 -> upsample to 64 x 64
  project stage 4             256 x   8 x   8 -> upsample to 64 x 64
    |
    v
  concatenate                1024 x  64 x  64
  fuse                         256 x  64 x  64
  classifier                     2 x  64 x  64
    |
    | bilinear upsample for loss/evaluation
    v
Output logits                   2 x 256 x 256
```

### SegFormer Shape Table

| Stage | Operation | Output shape |
|---|---|---|
| Input | image | `3 x 256 x 256` |
| Encoder stage 1 | patch embed + transformer blocks | `32 x 64 x 64` |
| Encoder stage 2 | patch embed + transformer blocks | `64 x 32 x 32` |
| Encoder stage 3 | patch embed + transformer blocks | `160 x 16 x 16` |
| Encoder stage 4 | patch embed + transformer blocks | `256 x 8 x 8` |
| Decode projection 1 | linear projection | `256 x 64 x 64` |
| Decode projection 2 | linear projection + upsample | `256 x 64 x 64` |
| Decode projection 3 | linear projection + upsample | `256 x 64 x 64` |
| Decode projection 4 | linear projection + upsample | `256 x 64 x 64` |
| Decoder concat | concatenate 4 scales | `1024 x 64 x 64` |
| Decoder fuse | 1x1 conv/fusion | `256 x 64 x 64` |
| Classifier | 1x1 conv | `2 x 64 x 64` |
| Final resize | bilinear upsample | `2 x 256 x 256` |

## Key Difference

```text
U-Net:
  Keeps high-resolution CNN features and uses skip connections to recover detail.

SegFormer:
  Builds multi-scale transformer features and fuses global/contextual features
  in a decoder, then upsamples to the mask size.
```

# Educational Semantic Segmentation on PASCAL VOC 2012

This project is a presentation-ready walkthrough of how semantic segmentation
evolved from hand-crafted computer vision to CNNs and then to transformer-based
models.

The goal is not only peak performance. The goal is to make the differences
visible, measurable, and easy to explain.

## What Is Semantic Segmentation?

Semantic segmentation assigns a class label to every pixel in an image. Instead
of saying "there is a dog somewhere in this image", a segmentation model says
"these exact pixels are dog, these pixels are person, these pixels are
background", and so on.

PASCAL VOC 2012 contains 21 segmentation labels:

- background
- 20 object classes such as person, dog, car, bus, cat, chair, and train

Pixels marked `255` are void pixels and are ignored during metric computation.

## Dataset Note

VOC 2012 publicly provides semantic segmentation masks for the official train
and validation splits. The official test labels are hidden on the evaluation
server, so this educational project creates:

- `train`: official VOC 2012 train split
- `val`: first deterministic half of official VOC 2012 val split
- `test`: second deterministic half of official VOC 2012 val split

This gives us labeled train, validation, and local test metrics for a live demo.

## Project Structure

```text
.
├── configs/              # YAML experiment settings
├── datasets/             # VOC download, split, dataloaders, sample views
├── traditional_cv/       # Sobel + morphology + watershed baseline
├── unet/                 # From-scratch PyTorch U-Net
├── transformer/          # SegFormer finetuning with HuggingFace
├── evaluation/           # metrics and final method comparison
├── visualizations/       # mask coloring, grids, plots
├── utils/                # config, seed, VOC class metadata
├── outputs/              # generated metrics, figures, checkpoints
└── notebooks/            # suggested presentation flow
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare and inspect the dataset:

```bash
.venv/bin/python3 -m datasets.prepare_dataset
.venv/bin/python3 -m datasets.visualize_samples
```

The dataset will be downloaded under `data/`.

## Faster Dataset Option

If the VOC mirror is too slow, use Oxford-IIIT Pet segmentation instead:

```bash
SEG_CONFIG=configs/oxford_pet.yaml .venv/bin/python3 -m datasets.prepare_dataset
SEG_CONFIG=configs/oxford_pet.yaml .venv/bin/python3 -m datasets.visualize_samples
SEG_CONFIG=configs/oxford_pet.yaml .venv/bin/python3 -m unet.train
```

Oxford Pet is a smaller educational segmentation dataset. It uses trimaps, which
this project maps to:

- `0`: background
- `1`: pet
- `255`: border/ignore

This is not a VOC replacement for 21-class benchmarking, but it is much faster
for a live local demo of the same segmentation pipeline.

## Method 1: Traditional Computer Vision

The classical baseline intentionally uses a simple, brittle pipeline:

1. Convert the image to grayscale.
2. Blur the image to reduce noise.
3. Use Sobel filters to detect edges.
4. Threshold the edge map.
5. Use morphology and connected components.
6. Use watershed to produce regions.
7. Convert discovered regions into a crude semantic mask.

Run it:

```bash
python -m traditional_cv.classical_pipeline
```

Outputs:

- intermediate images in `outputs/traditional_cv/`
- metrics in `outputs/metrics/traditional_cv_metrics.csv`

This method should fail in obvious ways. That is the point. Traditional methods
use hand-crafted features and local image cues. They do not understand object
identity, context, or class semantics.

## Why Traditional Methods Fail

Edges are not objects. A dog with soft fur, a person in shadows, and a chair
with thin legs can all break simple edge and region logic. Classical pipelines
often depend on fragile thresholds and assumptions about lighting, texture,
scale, and contrast.

This makes them useful as a teaching baseline:

- easy to explain
- visually interpretable
- poor semantic understanding
- brittle generalization

## Method 2: U-Net

U-Net is a CNN encoder-decoder architecture. The encoder compresses the image
into feature maps with increasing semantic meaning. The decoder upsamples those
features back to pixel resolution.

The key idea is the skip connection. Skip connections copy high-resolution
features from the encoder into the decoder. This helps the model recover object
boundaries and spatial detail after downsampling.

Train and evaluate:

```bash
python -m unet.train
```

Outputs:

- checkpoint at `outputs/checkpoints/unet_best.pt`
- metrics at `outputs/metrics/unet_metrics.csv`
- training history at `outputs/metrics/unet_history.csv`
- curves at `outputs/figures/unet_training_curves.png`
- predictions at `outputs/unet/examples/`

U-Net demonstrates why deep learning changed segmentation:

- features are learned automatically
- spatial hierarchy emerges from data
- skip connections combine detail with semantics
- predictions become class-aware, not just edge-aware

## Method 3: Transformer-Based Segmentation

This project uses SegFormer, a compact transformer-based segmentation model.
SegFormer uses hierarchical transformer features and a lightweight decoder.

Train and evaluate:

```bash
python -m transformer.train_segformer
```

Outputs:

- checkpoint folder at `outputs/checkpoints/segformer_voc/`
- metrics at `outputs/metrics/segformer_metrics.csv`
- training history at `outputs/metrics/segformer_history.csv`
- curves at `outputs/figures/segformer_training_curves.png`
- predictions at `outputs/segformer/examples/`

Transformers help segmentation because attention can model broader context.
For example, pixels that are far apart can still influence each other. This is
useful when local texture is ambiguous but global scene context is informative.

## Metrics

Every method is evaluated on train, validation, and local test splits using:

- pixel accuracy
- mean IoU
- Dice score

Pixel accuracy measures how many valid pixels were classified correctly. It is
easy to understand but can be dominated by background.

Mean IoU measures overlap between prediction and ground truth class by class.
It is the most common semantic segmentation metric.

Dice score measures overlap with extra emphasis on matching foreground regions.

## Final Comparison

After running the method scripts, generate the comparison outputs:

```bash
python -m evaluation.compare_methods
```

Outputs:

- `outputs/metrics/all_methods_metrics.csv`
- `outputs/metrics/all_methods_metrics.md`
- `outputs/figures/iou_comparison.png`
- `outputs/figures/qualitative_comparison_grid.png`

The qualitative grid shows the same images with:

1. original image
2. ground truth
3. traditional CV prediction
4. U-Net prediction
5. SegFormer prediction

## Suggested Live Demo Flow

1. Show VOC images and masks.
2. Run the traditional CV baseline and inspect intermediate edge/watershed images.
3. Explain why edge regions are not semantic classes.
4. Show U-Net architecture in `unet/model.py`.
5. Train U-Net briefly and show the training curves.
6. Show SegFormer training and discuss global context.
7. Generate the final comparison grid and metrics table.
8. Discuss why each generation improved on the previous one.

## Practical Notes

Default settings are intentionally modest so the project is approachable. For a
stronger demo, increase `epochs`, `image_size`, and `batch_size` in
`configs/default.yaml` according to your GPU memory.

The traditional CV method has a `max_images_per_split` limit because it exists
for demonstration, not serious training.

## Big Picture Conclusion

Traditional CV segmentation is hand-crafted and brittle. It can identify edges
and regions, but it does not know what objects are.

U-Net learns visual features from data and combines semantic abstraction with
spatial detail through skip connections.

Transformer-based segmentation adds stronger long-range context modeling,
helping the model reason about complete objects and scenes rather than only
local patterns.

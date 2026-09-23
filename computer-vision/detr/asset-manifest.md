# DETR Paper Asset Manifest

Primary source: Nicolas Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020.

Source paper: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

Rendering method: PyMuPDF rasterization at 5× source resolution with explicit PDF-coordinate crop rectangles. No complete PDF pages are embedded in the presentation.

## Architecture figure

### Asset: `assets/detr-architecture.png`

- Source paper: End-to-End Object Detection with Transformers
- PDF page: 7
- Reference: Figure 2
- Crop coordinates: `(32, 32, 380, 126)` PDF points
- Output size: 1740×470 PNG
- Usage: Slides 6–11
- Modification: Exact figure crop. Slides 7–11 add external colored outline highlights; the underlying figure pixels and original labels are unchanged.
- Citation: `Source: Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020, Fig. 2.`

## Optimal assignment equation

### Asset: `assets/optimal-assignment.png`

- Source paper: End-to-End Object Detection with Transformers
- PDF page: 5
- Reference: Equation (1)
- Crop coordinates: `(112, 228, 386, 276)` PDF points
- Output size: 1370×240 PNG
- Usage: Slide 16
- Modification: Exact high-resolution crop with whitespace retained around the equation and its original equation number.
- Citation: `Source: Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020, Eq. (1).`

## Hungarian training-loss equation

### Asset: `assets/hungarian-loss.png`

- Source paper: End-to-End Object Detection with Transformers
- PDF page: 5
- Reference: Equation (2)
- Crop coordinates: `(55, 512, 386, 557)` PDF points
- Output size: 1655×225 PNG
- Usage: Slide 26
- Modification: Exact high-resolution crop with the original equation number.
- Citation: `Source: Carion et al., “End-to-End Object Detection with Transformers,” ECCV 2020, Eq. (2).`

## HTML equation reproductions

The following teaching equations are rendered as selectable KaTeX HTML overlays. KaTeX JavaScript, CSS, and fonts are embedded into the standalone HTML file; no network access or original PDF access is required.

- Slide 17: Exact symbolic reproduction of Equation (1).
- Slide 19: Adapted pairwise matching-cost definition for a real target.
- Slide 20: Adapted classification component from the unnumbered matching-cost definition.
- Slide 21: Adapted L1 component from the unnumbered bounding-box loss.
- Slide 22: Adapted generalized-IoU component from the unnumbered bounding-box loss.
- Slide 28: Adapted combined box-loss summary using GIoU and L1.
- Slide 30: Exact symbolic reproduction of Equation (2).

Each teaching slide states either `Reproduced from` or `Adapted from` according to whether the displayed formula preserves the paper equation exactly.

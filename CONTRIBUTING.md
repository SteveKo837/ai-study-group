# Contributing

## Contribution workflow

Use a dedicated branch and Pull Request for every change; do not commit directly
to `main`.

```text
main → create branch → add or update materials → update README navigation
     → commit → push → open PR → review → merge
```

Start from an up-to-date `main`. Review your changes and verify links and assets
before committing, then push your branch and open a PR targeting `main`.

### Branch naming

Use `docs/add-<topic>`, `docs/update-<topic>`, or `docs/fix-<topic>`, for example:

```text
docs/add-image-segmentation
docs/add-ppo
docs/update-flash-attention
docs/fix-moe-images
```

### Commit messages

Use concise messages describing the change:

```text
docs: add image segmentation material
docs: update PPO presentation
docs: add MoE architecture diagrams
docs: fix broken presentation links
```

## Material structure and naming

Organize materials by content under `<domain>/<topic>/`, not by author or date.
Create a domain only when adding material that belongs in it. A typical HTML
presentation has this structure:

```text
computer-vision/
├── README.md
└── image-segmentation/
    ├── README.md
    ├── index.html
    └── assets/
        ├── architecture.png
        ├── example.png
        └── workflow.svg
```

Use lowercase kebab-case for directory and material filenames, such as
`reinforcement-learning`, `mixture-of-experts`, and `flash-attention`.
Keep conventional documentation filenames such as `README.md` and
`CONTRIBUTING.md`. Folder names must describe the subject and must not contain
author names, dates, or personal identifiers. Avoid names such as `steve-ppo`,
`2026-09-23-ppo`, `ppo-by-steve`, or `PPO_Final`.

Use `index.html` as the entry point for an HTML presentation and keep its images
and supporting files in `assets/`. Notes-only topics need not include HTML or an
empty assets directory. Preserve existing materials; avoid unnecessary renames
or moves, and update affected links whenever a path changes.

### Topic README

Each topic needs a short `README.md` with author, date (`YYYY-MM-DD`), domain,
overview, and relevant references. Link to the actual material so the README
serves as its entry point. For example:

```markdown
# Image Segmentation

**Author:** Steve  
**Date:** 2026-09-23  
**Domain:** Computer Vision

## Overview

An introduction to image segmentation methods and their applications.

[Open the presentation](./index.html)

## References

- Add the papers, documentation, or other sources used, with links.
```

Replace the example metadata and references with accurate information. For
notes or other formats, link to those files instead of `index.html`.

### HTML and assets

All HTML resource paths (images, stylesheets, scripts, and other supporting
files) must be relative to the file that uses them. For example:

```html
<img src="./assets/architecture.png" alt="Model architecture">
```

Do not use machine-specific paths such as `C:/Users/.../architecture.png`,
`/Users/.../architecture.png`, or `file://` URLs, or site-root paths such as
`/assets/architecture.png`. External reference hyperlinks may use HTTPS URLs.
Check resource paths in CSS as well; those resolve relative to the stylesheet.

## README navigation updates

Keep the navigation hierarchy:

```text
Root README → domain README → topic directory → material
```

- When adding a topic, add or update its topic README and its row in the domain
  README. For `computer-vision/image-segmentation/`, update
  `computer-vision/README.md`.
- When adding a new top-level domain, also add a link to its directory in the
  root `README.md`, using the domain name as the label and a relative directory
  path such as `./computer-vision/` as the target.
- Keep the root index limited to major domains. Do not list individual topics
  there or add empty categories.
- When updating or moving materials, keep index metadata and affected links in
  sync. Use relative links with exact filename capitalization so they work on
  GitHub.

Each domain README should index its materials using this format, with topic
links relative to the domain directory:

```markdown
# Computer Vision

## Materials

| Topic | Title | Author | Date |
|---|---|---|---|
| [Image Segmentation](./image-segmentation/) | Image Segmentation Overview | Steve | 2026-09-23 |
```

## Pull Request requirements

Explain the topic being added or updated, the files or directories changed,
whether existing content is affected, and which README indexes were updated.
For example:

```markdown
## Summary

Add study materials for Image Segmentation. Existing materials are unaffected.

## Changes

- Add `index.html`, supporting diagrams in `assets/`, and a topic `README.md`
- Update `computer-vision/README.md` to index the new topic
- Update root `README.md` if Computer Vision is a new domain

## Path

computer-vision/image-segmentation/

## Validation

- Opened the presentation and checked images and navigation
- Verified topic, domain, and root README links
```

### PR checklist

Include this checklist in your PR and mark non-applicable items as `N/A` with a
brief explanation.

- [ ] Changes are made on a dedicated branch
- [ ] Materials are placed under the correct domain
- [ ] Folder and material filenames use lowercase kebab-case where appropriate
- [ ] Folder names do not contain author names, dates, or personal identifiers
- [ ] `index.html` opens correctly, if included
- [ ] Images and assets load correctly
- [ ] HTML and supporting asset resource paths are relative
- [ ] Topic `README.md` contains author/date/domain metadata and material links
- [ ] Domain README indexes the new or updated topic
- [ ] Root README links to the domain if needed
- [ ] README links have been verified
- [ ] No temporary or unnecessary large files are committed

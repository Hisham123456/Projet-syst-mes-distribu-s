# Pipeline diagram

TODO(A1.9): draw the CPU baseline pipeline (Mermaid, ASCII or an image linked from here).
One box per stage: loading, preprocessing, filters, augmentation, forward/backward,
optimizer step, evaluation. Between boxes write the tensor shape and dtype (for example
`N x 224 x 224 x 3 uint8`). Mark which stages run once and which run every epoch.
Later, add a second diagram of your optimized pipeline that shows where data crosses the
CPU/GPU boundary.

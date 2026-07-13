# Product detector training

This project can train a single-class YOLO detector from collected crops in
`dataset_output`. Every object is labeled as:

```yaml
0: product
```

## Dataset format

The collector currently saves object crops, not full source frames with manual
bounding boxes. `train_product_detector.py` converts every crop image into one
YOLO sample and writes a label that covers almost the full crop:

```txt
0 0.5 0.5 0.98 0.98
```

The generated dataset is written to:

```txt
dataset_output/yolo_product_dataset
```

## Run training

From the repository root:

```bash
.venv/bin/python train_product_detector.py
```

Useful options:

```bash
.venv/bin/python train_product_detector.py \
  --source-dir dataset_output \
  --model yolov8n.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --name product
```

To only prepare the YOLO dataset without training:

```bash
.venv/bin/python train_product_detector.py --prepare-only
```

Training outputs are saved under:

```txt
runs/product_detector/product
```

The default `yolov8n.yaml` trains a detector from scratch and does not require
downloading a pretrained checkpoint. If you have a local detection checkpoint,
pass it with `--model path/to/detector.pt`.

The best checkpoint is usually:

```txt
runs/product_detector/product/weights/best.pt
```

## Use the trained model

After training, set the scan model path in `config.py`:

```python
SCAN_MODEL_PATH = Path("runs/product_detector/product/weights/best.pt")
```

For a normal YOLO detector, the runtime should use `YOLO`/`track` instead of
`YOLOWorld.set_classes(...)`. The trained class name is already `product`.

## Notes

This dataset teaches the model that each crop contains one product. It is useful
for making a product-present detector, but full-frame detection quality is better
when training data contains original frames and manually checked product boxes.

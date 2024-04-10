
# Research Paper To-Do List

Data type:

- big: original dataset
- small:
  - train -> gt_labels
  - clutter -> filtered with obj
  - test -> if_net
  - clutter -> filtered with obj 

## Ablation study

### Study the effect of the database difference for different backbones

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| EfficientNet - big dataset - 50    | -      |  -   |  Sebastian | Training - ws-l6-001 |
| EfficientNet - small dataset - 50  | -      |  -   | Unknown | Unknown |
| Phinet - big dataset - 50    | -      |  -   | Kami | dataset download -ws-l6-017 |
| Phinet - small dataset - 50  | -      |  -   | Unknown | Unknown |
| MobileVIT - big dataset - 50    | -      |  -   | Unknown | Unknown |
| MobileVIT - small dataset - 50  | -      |  -   | Unknown | Unknown |

Good images:
![good](./docs/images/good_images.png "Good images")

Bad images:
![bad](./docs/images/bad_images.png "Bad images")

### Study the effect in the change in the number of epochs for different backbones 

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| EfficientNet - 50    | -      |  -   | Unknown | Unknown |
| EfficientNet - 500  | -      |  -   | Sebastian | Testing - kami02 |
| Phinet - 50    | -      |  -   | Unknown | Unknown |
| Phinet - 500  | -      |  -   | Unknown | Unknown |
| MobileVIT - 50    | -      |  -   | Unknown | Unknown |
| MobileVIT - 500  | -      |  -   | Brana | ws-l3-001 |

### Study of the effect of self supervised pretraining on the downstream task (initial investigation)

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| Phinet - 50    | -      |  -   | Unknown | Unknown |
| Phinet - 500  | -      |  -   | Unknown | Unknown |

## Some tricks

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| Phinet - 50 - Postprocessing   | -      |  -   | Unknown | Unknown |
| Phinet - 50 - Majority voting   | -      |  -   | Unknown | Unknown |
| Phinet - 50 - Something else?   | -      |  -   | Unknown | Unknown |





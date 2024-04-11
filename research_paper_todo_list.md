
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
| EfficientNet - big dataset - 50    | 0.6519797086743283 | 0.03974051029645851 |  Sebastian | Done |
| EfficientNet - small dataset - 50  | -      |  -   | Sebastian | Training - ws-l5-008 |
| EfficientNet - big dataset - 500  |  0.6781074918554808 | 0.0386619883365546  | Sebastian | Done - Kami02 |
| EfficientNet - small dataset - 500  | 0.6704734775094003  | 0.03857772670822  | Sebastian | Done - Kami02 |
| Phinet - big dataset - 50    | -      |  -   | Kami | Testing - ws-l6-010 |
| Phinet - small dataset - 50  | -      |  -   | Unknown | Unknown |
| MobileVIT - big dataset - 50    | -      |  -   | Unknown | Unknown |
| MobileVIT - small dataset - 50  | -      |  -   | Unknown | Unknown |


### Study the effect in the change in the number of epochs for different backbones (small-dataset) --> PROBABLY WILL CHANGE BIG DATASET

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| EfficientNet - 50    | -      |  -   | Unknown | Unknown |
| EfficientNet - 500  | 0.6704734775094003  | 0.03857772670822  | Sebastian | Done |
| Phinet - 50    | -      |  -   | Unknown | Unknown |
| Phinet - 500  | -      |  -   | Sebastian | Training - Kami02 |
| MobileVIT - 50    | -      |  -   | Brana |  ws-l3-001 |
| MobileVIT - 500  | -      |  -   | Unkown | Unkown |

### Study of the effect of self-supervised pretraining on the downstream task (initial investigation)

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


TODO:

- [ ] take away the folders with less than 50 images, to exclude them from testing
- [ ] fix macs calculation to export correctly to json file

NOTE:
we can also write about the images that contain both objects that could both belong to some personalization objects -> AMBIGUITIES

Good images:
![good](./docs/images/good_images.png "Good images")

Bad images:
![bad](./docs/images/bad_images.png "Bad images")


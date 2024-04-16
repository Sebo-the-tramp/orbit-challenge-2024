
# Research Paper To-Do List

## Ablation study

### Study the effect of the database difference for different backbones

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| EfficientNet - big dataset - 50    | 0.6519797086743283 | 0.03974051029645851 |  Sebastian | Done |
| EfficientNet - small dataset - 50  | 0.6530725011837245 | 0.03935080769579349 | Sebastian | Done - ws-l5-008 |
| EfficientNet - big dataset - 500  |  0.6781074918554808 | 0.0386619883365546  | Sebastian | Done - Kami02 |
| EfficientNet - small dataset - 500  | 0.6704734775094003  | 0.03857772670822  | Sebastian | Done - Kami02 |
| Phinet - big dataset - 50    | 0.5969983820930811      |  0.04065243122345036   | Brana | Done |
| Phinet - small dataset - 50  | 0.6319738170536263     |  0.04035598926922223 | Sebastian | Done - ws-l5-008  |
| Mobilenet V3 - big dataset - 50    | 0.5763458436156461| 0.042366017120875704 | Kami |Done|
| Mobilenet V3 - small dataset - 50  | 0.6591124930687723| 0.0395277036588839 | Kami| Done|

[//]: # (| Phinet - big dataset - 50 - 1 epoch  | 0.6045990629830172      |  0.04094082216687125   | Sebastian | Done |)
[//]: # ( | Phinet - small dataset - 50 max support sampler instead of random  | 0.62288852918077     | 0.04046261139498393 | Sebastian | Done - ws-l5-008  |)

Description of the dataset types:
- *big*: original
- *medium*: **dropped:** 'object_not_present_issue', 'occlusion_issue', 'blur_issue' + IFnetwork in test clean 
- *small*: only the best images in the dataset + IFnetwork in test clean 

### On phinet 50 epochs
 
| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| Phinet - big dataset - 50  (1.8 mil. images)  | 0.5969983820930811      |  0.04065243122345036   | Brana | Done |
| Phinet - medium dataset - 50 (1.3 mil. images) | 0.6195229377169459     | 0.04087509739856873 | Kami | Done |
| Phinet - small dataset - 50 (1.1 mil. images) |  0.6319738170536263     |   0.04035598926922223  | Sebastian | Done  |

### On phinet 500 epochs

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| Phinet - big dataset - 500  (1.8 mil. images)   |  0.6083022641687181       |  0.040819703094845325   | Brana | Done |
| Phinet - medium dataset - 500 (1.3 mil. images) | -    | - | Unknown | Training (?) |
| Phinet - small dataset - 500 (1.1 mil. images)  | -    | - | Unknown | Unknown |


### Study the effect in the change in the number of EPISODES for different backbones (big db)

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| EfficientNet - 50  (big db) | 0.6519797086743283 | 0.03974051029645851 |  Sebastian | Done |
| EfficientNet - 500 (big db) | 0.6704734775094003  | 0.03857772670822  | Sebastian | Done |
| Phinet - 50 (big db)   | 0.5969983820930811      |  0.04065243122345036   | Brana | Done |
| Phinet - 500 (big db)  | 0.6083022641687181      |  0.040819703094845325  | Sebastian | Done - Kami02 |
| Mobilenet V3 - 50 (big db)   |0.5763458436156461| 0.042366017120875704  | Kami | Done |
| Mobilenet V3 - 500 (big db)  | 0.5894863208652916 | 0.04112786544328679 | Brana | Done |

### Study the effect in the change in the number of EPOCHS for phinet (small db)

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
|  Phinet - 50 - 2 epochs  | 0.6319738170536263     |  0.04035598926922223 | Sebastian | Done |
|  Phinet - 50 - 5 epochs  | 0.629246002259729    |  0.040353474313382216| Kami | Done  |

### Study of the effect of self-supervised pretraining on the downstream task (initial investigation)

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status | NOTE |
|--------------------------|--------------------------|-----------------------|------|--------|------|
| Phinet - 50 - Imagenet 30% acc | 0.35792436117508347 | 0.04466637960245303  | Sebastian | Done |  |
| Phinet - 50 - Cifar 50% acc | 0.2960068629004077  |  0.040999001825241445  | Sebastian | Done ||
| Phinet - 500 - Imagenet 20% acc | -      |  -   | INCOMINNNG | Sebastian | NOT POSSIBLE FOR TIME |
| Phinet - 500 - Cifar 50% acc | -      |  -   | Pending GPU | Sebastian - ws-l6-007 | BUG IN THE CODE|

## Some tricks

| Name                     |  Average Frame Accuracy  | Confidence Interval   |  WHO | Status |
|--------------------------|--------------------------|-----------------------|------|--------|
| Phinet - 50 - Postprocessing   | -      |  -   | Unknown | Unknown |
| Phinet - 50 - Majority voting   | -      |  -   | Unknown | Unknown |
| Phinet - 50 - Self supervised at this point  | -      |  -   | Unknown | Unknown |


TODO:

- [ ] take away the folders with less than 50 images, to exclude them from testing
- [ ] fix macs calculation to export correctly to json file

NOTE:
we can also write about the images that contain both objects that could both belong to some personalization objects -> AMBIGUITIES

Good images:
![good](./docs/images/good_images.png "Good images")

Bad images:
![bad](./docs/images/bad_images.png "Bad images")


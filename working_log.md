# Working Log

So the most interesting things are happening in the /module/protonet_video_post file. There we have canny Edge detector, that could be easily substituted by the backbone, we are going to use. Because anyway the mac to personalize might be even less if we share the backbone.di

### 03/04/2024
So today we were able to do the training/testing on the full dataset, use the follwing code to get enough RAM on A6000, the limit for the 4090 is 44GB:

```bash 
    salloc --x11 -N1 -n12 -w $1 --mem=100G
```

We are trying to replace the backbone:
- mobilevitv2 - 0.75
- phinetBIG 

Let's see. 

Also refactoring of the github repository. It needs to be cleaned.

- Added IFresV4, the log for saving the images that are worth for the support vectors

## Hackathon metamadness

### 05/04/2024 - 1st day hackathon

We got some initial results:

| Name                     |  Average Frame Accuracy  | Confidence Interval   |
|--------------------------|--------------------------|-----------------------|
| EfficientNet - 50 episodes - big dataset (BASELINE)   | 0.6638479368843097       | ±0.05643283613519739   |
| Phinet - 50 episodes - big dataset   | 0.6509896065430318      |  ±0.05423007214999352 |
| Phinet - 50 episodes - better dataset (non-legit)  | 0.6803234484591006      |  ±0.05423007214999352 |


### 06/04/2024 - 2nd day hackathon

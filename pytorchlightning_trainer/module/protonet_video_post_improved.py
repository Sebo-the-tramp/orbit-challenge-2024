from pathlib import Path

import torch
from torch.nn import functional as F
import torchmetrics

import gc

from src.official_orbit.utils.data import attach_frame_history
from src.utils.evaluator import FEATEpisodeEvaluator
from src.data.orbit_few_shot_video_classification import \
    get_object_category_names_according_labels
from pytorchlightning_trainer.module.protonet import ProtoNetWithLITE
from src.utils.missing_object_detector import filter_support_clips_without_object

class ProtoNetWithLITEVideoPostImproved(ProtoNetWithLITE):
    """
        This class implements ProtoNet using LITE, also integrated with the video support clips selection
        using Canny Edge Detector during testing.

    """

    # def on_test_batch_start(self, batch, batch_idx, _) -> None:
    #     print("THIS SHOULD BE CALLED BEFORE THE DATALOAED LOADS ONE USER")

    #     # Hard code; To avoid direct modifications on official_orbit.models.few_shot_recognizers.py
    #     self.model._set_device(self.device)
    #     self.model.set_test_mode(True)
    #     self.episode_evaluator = FEATEpisodeEvaluator(  # feat or not feat here
    #         save_dir=str(Path(self.trainer.logger.root_dir, "testing_per_video_results")))


    # def on_test_batch_end(self, batch, batch_idx, _, _1) -> None:
    #     print("THIS SHOULD BE CALLED AFTER THE DATALOAED LOADS ONE USER")   

    #     self.episode_evaluator.compute_statistics()
    #     self.episode_evaluator.save_to_disk()
    #     self.episode_evaluator.reset()

    #     self.model._reset()


    # Could be that I could just rewrite the test_step, but the problem it seems that when test_step is called the objects images should already be on the GPU

    def on_test_start(self) -> None:
        # print(torch.cuda.memory_summary())        

        self.model._set_device(self.device)
        self.model.set_test_mode(True)
        self.episode_evaluator = FEATEpisodeEvaluator(
            save_dir=str(Path(self.trainer.logger.root_dir, "testing_per_video_results")))
        
        # print("test start", torch.cuda.memory_summary())        

    def test_step(self, val_batch, batch_idx):

        print(val_batch["user_id"])

        support_clips_frames = val_batch['support_frames']
        support_clips_labels = val_batch['support_labels']
        support_clips_filenames = val_batch['support_frame_filenames']

        # print(next(iter(val_batch.values())).device)

        # print("BIMBOOOOO", len(support_clips_frames))
        # print("asdasdasd", len(val_batch['query_frame_filenames'][0]))

        # use the support images to create protonets -> maccs to personalize len(support_clips_frames) * phinet macc forward
        # easy

        #but then were are the query?


        ### For every batch we get N objects and 200 frames per object. OK?
        # So if the number of N *200 is too big, this cannot be allocated in memory right?

        # What if we do a for loop on N and we set a limit on the number of images that can fit in the memory?

        ## Like 200 * 224 * 224 * 3 / 8 / 1024 -> around 37 MB?
        ## Since we have 24 GB, we might be able to allocate up to 5 objects for each loop. if my calc are correct.

        # print("BEGINNING\n")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        object_category_names = \
            get_object_category_names_according_labels(val_batch['query_frame_filenames'], val_batch['query_labels'])
        
        # for x in range(len(val_batch['query_frame_filenames'])):
        #     print(len(val_batch['query_frame_filenames'][x]))
        
        self.episode_evaluator.register_object_category(object_category_names)

        self.model.personalise(support_clips_frames, support_clips_labels)

        # print("BEFORE\n")
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    
        # print("AFTER\n")

        support_clips_frames.to('cpu').detach()


        print("TOTAL NUMBER OF VIDEOS", len(val_batch['query_labels']))

        print("TOTAL NUMBER OF IMAGES USED", len(val_batch['query_labels']) * 200 )

        print("TOTAL SIZE OF IMAGES USED", len(val_batch['query_labels']) * 200 * 224 * 224 * 3 / 1024 / 1024, "MB")        


        ### HERE I SHOULD ALREADY BE ABLE TO RELEASE THE SUPPORT CLIP FRAMES
        del support_clips_frames, support_clips_labels, support_clips_filenames
        gc.collect()  # Force the garbage collector to run
        torch.cuda.empty_cache()  # Clear cache after deleting
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))        

        for video_sequence_frames, video_sequence_label, video_frame_filenames in \
                zip(val_batch['query_frames'], val_batch['query_labels'], val_batch['query_frame_filenames']):
            
            # print("yess here going crazy")
            # I am not sure what this does, but it fucks up many things
            # video_clips_frames = attach_frame_history(video_sequence_frames, self.video_clip_length)


            # print("BACKWARD 2:", video_sequence_frames.shape)
            video_logits, video_features = self.model.predict(video_sequence_frames)  # Shape = [num_frames, num_classes]
            print(video_logits.shape)
            video_prediction_scores = F.softmax(video_logits, dim=-1)
            video_predictions = video_logits.argmax(dim=-1).detach().cpu()
            num_frames = video_logits.shape[0]
            video_labels = video_sequence_label.expand(num_frames).detach().cpu()
            acc = torchmetrics.functional.accuracy(video_predictions, video_labels)
            self.episode_evaluator.add_video_result(per_frame_prediction_scores=video_prediction_scores.detach().cpu().numpy(),
                                                    per_frame_features=video_features.detach().cpu().numpy(),
                                                    frame_filenames=video_frame_filenames,
                                                    video_gt_label=video_sequence_label.item(),
                                                    video_frame_accuracy=acc.item())


        self.model._reset()

        # print(" end", torch.cuda.memory_summary())

        self.episode_evaluator.compute_statistics()
        self.episode_evaluator.save_to_disk()
        self.episode_evaluator.reset()

        # Example of deleting specific tensors
        # del video_clips_frames, video_logits, video_features, video_predictions, video_prediction_scores, video_labels
        # # del support_clips_frames, support_clips_labels, support_clips_filenames, num_valid_support_clips
        # # del prototypes, num_total_support_clips, object_category_names
        # del val_batch
        # gc.collect()  # Force the garbage collector to run
        # torch.cuda.empty_cache()  # Clear cache after deleting

        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print("FINEEE EPISODIO ")
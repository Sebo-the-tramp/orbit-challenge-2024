BEGIN=$(date +"%H:%M:%S")

python run.py data=test_support_sampler_uniform_fixed_chunk_size_10 model=feat_with_lite_video_post_improved train=with_lite_test train.exp_name="50_efficientnet_2checkpoint"
#python run.py data=default_use_data_aug model=feat_with_lite_video_post train=with_lite_test train.exp_name="reproduce_our_testing_result"

echo "Current date: $BEGIN"

ENDIN=$(date +"%H:%M:%S")

curl -X POST \
     -H 'Content-Type: application/json' \
     -d "{\"chat_id\": \"-1002101713030\", \"text\": \"Either the training finished or something went wrong. PHINET SMALL 50 episodes. Begin: $BEGIN, End: $ENDIN\", \"disable_notification\": true}" \
     https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage

BEGIN=$(date +"%H:%M:%S")

python run.py data=default_use_data_aug model=feat_with_lite train=with_lite_train train.exp_name="efficientNet_50_big" 

echo "Current date: $BEGIN"

ENDIN=$(date +"%H:%M:%S")

curl -X POST \
     -H 'Content-Type: application/json' \
     -d "{\"chat_id\": \"-1002101713030\", \"text\": \"Either the training finished or something went wrong. PHINET SMALL 50 episodes. Begin: $BEGIN, End: $ENDIN\", \"disable_notification\": true}" \
     https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage
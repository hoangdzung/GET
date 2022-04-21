python MasterFC/master_mac.py --dataset="Snopes" \
                             --cuda=1 \
                             --fixed_length_left=30 \
                             --fixed_length_right=100 \
                             --log="logs/get" \
                             --loss_type="cross_entropy" \
                             --batch_size=32 \
                             --num_folds=5 \
                             --use_claim_source=0 \
                             --use_article_source=1 \
                             --path="formatted_data/declare/" \
                             --hidden_size=300 \
                             --epochs=100 \
                             --num_att_heads_for_words=5 \
                             --num_att_heads_for_evds=2 \
                             --gnn_window_size=3 \
                             --lr=0.0001 \
                             --gnn_dropout=0.2 \
                             --seed=123756 \
                             --gsl_rate=0.6

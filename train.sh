
# Datasets
family_dataset_path=$PWD'/KGs/Family/family-benchmark_rich_background.owl'

# Embeddings
family_kge=$PWD'/embeddings/ConEx_Family/ConEx_entity_embeddings.csv'

num_episode=100
min_num_concepts=2
num_of_randomly_created_problems_per_concept=1
relearn_ratio=5
echo "Training Starts"
python drill_train.py --path_knowledge_base "$family_dataset_path" --path_knowledge_base_embeddings "$family_kge" --num_episode $num_episode --min_num_concepts $min_num_concepts --num_of_randomly_created_problems_per_concept $num_of_randomly_created_problems_per_concept --relearn_ratio $relearn_ratio --use_illustrations False
echo "Training Ends"
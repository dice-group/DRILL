
# Datasets
path_family_dataset=$PWD'/KGs/Family/family-benchmark_rich_background.owl'
# Embeddings
path_family_kge=$PWD'/embeddings/ConEx_Family/ConEx_entity_embeddings.csv'
# Pretrained Models
path_drill_family=$PWD'/Log/20211213_103505_209642/DrillHeuristic_averaging.pth'

# Benchmark Learning Problems
standard_family_benchmark_lp_path=$PWD'/LPs/Family/lp_dl_learner.json'
# LPs with concepts to be ignored
family_benchmark_lp_w_injection_path=$PWD'/LPs/Family/lp_dl_learner_with_injection.json'
# LPs with no negatives
family_benchmark_lp_wo_negatives_path=$PWD'/LPs/Family/lp_dl_learner_with_pos_only.json'


echo "#####################################################################"
echo "Start Testing on Family on learning problems obtained from DL-learner"
echo "#####################################################################"
python experiments_standard.py --path_lp $standard_family_benchmark_lp_path --path_knowledge_base $path_family_dataset --path_knowledge_base_embeddings $path_family_kge --pretrained_drill_avg_path $path_drill_family
echo "#####################################################################"
echo "Start Testing on Family on learning problems with prior knowledge injection obtained from DL-learner"
echo "#####################################################################"

python experiments_standard.py --path_lp $family_benchmark_lp_w_injection_path --path_knowledge_base $path_family_dataset --path_knowledge_base_embeddings $path_family_kge --pretrained_drill_avg_path $path_drill_family
echo "#####################################################################"
echo "Start Testing on Family on learning problems without negative examples  obtained from DL-learner"
echo "#####################################################################"
python experiments_standard.py --path_lp $family_benchmark_lp_wo_negatives_path --path_knowledge_base $path_family_dataset --path_knowledge_base_embeddings $path_family_kge --pretrained_drill_avg_path $path_drill_family



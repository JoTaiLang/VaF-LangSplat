scenes=("figurines")

exp_name='VaF-langsplat'
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1
gpu=-1
feature_level=1


for scene in ${scenes[*]};
do
	bash train.sh -d Lerf/${scene} -l ${exp_name} --model_path output/${scene} --gpu ${gpu} --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} --ratio ${ratio} --feature_level ${feature_level}
done
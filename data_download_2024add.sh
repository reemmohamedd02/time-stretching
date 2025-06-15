mkdir -p "data/dcase2024t2/eval_data/raw"

# download eval data
cd "data/dcase2024t2/eval_data/raw"

# for machine_type in 3DPrinter AirCompressor Scanner ToyCircuit HoveringDrone HairDryer ToothBrush RoboticArm BrushlessMotor; do
# wget "https://zenodo.org/records/11183284/files/eval_data_${machine_type}_train.zip"
# unzip "eval_data_${machine_type}_train.zip"
# done

for machine_type in \
    HairDryer_train \
; do
curl -L -O "https://zenodo.org/records/11259435/files/eval_data_${machine_type}.zip"
unzip "eval_data_${machine_type}.zip"
done


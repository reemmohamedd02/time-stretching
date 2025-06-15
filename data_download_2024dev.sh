mkdir -p "data/dcase2024t2/dev_data/raw"

# download dev data
cd "data/dcase2024t2/dev_data/raw"
for machine_type in bearing fan gearbox slider ToyCar ToyTrain valve; do
curl -L -O "https://zenodo.org/record/10902294/files/dev_${machine_type}.zip"
unzip "dev_${machine_type}.zip"
done

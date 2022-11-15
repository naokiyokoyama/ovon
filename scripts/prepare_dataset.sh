valid=true

# Check for two args
if [ -z "$2" ]
then
    echo "Please provide two paths to hm3d-train-glb-v0.2 and hm3d-val-glb-v0.2 as args"
    valid=false
fi

# Check for valid paths
if [ ! -d "$1" ]
then
    echo "$1 is not a valid directory!"
    echo "Please provide a valid path for hm3d-train-glb-v0.2"
    valid=false
fi

if [ ! -d "$2" ]
then
    echo "$2 is not a valid directory!"
    echo "Please provide a valid path for hm3d-val-glb-v0.2"
    valid=false
fi

if [ $valid = true ]
then
    mkdir -p habitat-lab/data/scene_datasets/hm3d &&
    echo "Creating symlink from $1 to habitat-lab/data/scene_datasets/hm3d" &&
    ln -s `realpath $1` habitat-lab/data/scene_datasets/hm3d/train &&
    echo "Creating symlink from $2 to habitat-lab/data/scene_datasets/hm3d" &&
    ln -s `realpath $2` habitat-lab/data/scene_datasets/hm3d/val &&
    echo "Done"
fi

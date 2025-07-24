import h5py


f = h5py.File("/home/will/IsaacLab/dexterous/data/allegro_inhand_rollouts.hdf5", "r")

breakpoint()
print(f.keys())
print(f["mask/train"][:])
print(f["mask/test"][:])
f.close()
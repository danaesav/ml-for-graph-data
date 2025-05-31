import torch
import os
import os.path as osp
import numpy as np

def split_hyper_data(data_name="Hyperspheres_10_10_0", train_percent=0.2, val_percent=0.2, num_split=3, path=""):

    folder_name = f"{data_name}_{train_percent}"

    # the folder exists
    if os.path.exists(folder_name):
        # exist and not empty:
        if os.listdir(folder_name):
            print("the corresponding splits exist! Do not split again")
        # folder exists and but empty
        else:
            labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                            skip_header=1, dtype=np.dtype(float), delimiter=',')
            labels = torch.tensor(labels).float()
            num_nodes = labels.shape[0]

            # Create a tensor of node indices
            node_indices = torch.arange(num_nodes)

            # Calculate the sizes of the splits
            train_size = int(num_nodes * train_percent)
            val_size = int(num_nodes * val_percent)
            test_size = num_nodes - train_size - val_size

            for i in range(num_split):
                # Shuffle the node indices
                shuffled_indices = torch.randperm(num_nodes)

                # Split the node indices into training, validation, and test sets
                train_indices = shuffled_indices[:train_size]
                val_indices = shuffled_indices[train_size:train_size+val_size]
                test_indices = shuffled_indices[train_size+val_size:]

                print(train_indices.shape)
                print(val_indices.shape)
                print(test_indices.shape)

                # Save the splits
                torch.save({'train_mask': train_indices, 'val_mask': val_indices, 'test_mask': test_indices}, os.path.join(folder_name, 'split_'+str(i)+'.pt'))
            
    # does not exits, create folder and split data
    else:

        print('No splits of ', data_name, " with corresponding percentage exits, spliting the data now...")
            # Create the folder
        os.makedirs(folder_name)

        labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                            skip_header=1, dtype=np.dtype(float), delimiter=',')
        labels = torch.tensor(labels).float()
        num_nodes = labels.shape[0]

        # Create a tensor of node indices
        node_indices = torch.arange(num_nodes)

        # Calculate the sizes of the splits
        train_size = int(num_nodes * train_percent)
        val_size = int(num_nodes * val_percent)
        test_size = num_nodes - train_size - val_size

        for i in range(num_split):
            # Shuffle the node indices
            shuffled_indices = torch.randperm(num_nodes)

            # Split the node indices into training, validation, and test sets
            train_indices = shuffled_indices[:train_size]
            val_indices = shuffled_indices[train_size:train_size+val_size]
            test_indices = shuffled_indices[train_size+val_size:]

            print(train_indices.shape)
            print(val_indices.shape)
            print(test_indices.shape)

            # Save the splits
            torch.save({'train_mask': train_indices, 'val_mask': val_indices, 'test_mask': test_indices}, os.path.join(folder_name, 'split_'+str(i)+'.pt'))



split_hyper_data(data_name="Hyperspheres_10_10_0", train_percent=0.01, val_percent=0.2, num_split=3, path="")
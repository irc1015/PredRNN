__author__ = 'irccc'

import Moving_MNIST

datasets_map = {
    'mnist': Moving_MNIST
}

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size, img_height, img_width, seq_length,is_training=True):
    '''
    Input:
        dataset_name: name of dataset
        train_data_paths: may several paths
        valid_data_paths: may several paths
        batch_size: the number of sequences one time
        img_height:
        img_width:
        seq_length:
        injection_action:
        is_training: whether set training mode
    Determine:
        test_input_param dict --> test_input_handle
            paths, minibatch_size, input_data_type, is_output_sequence, name
        (if train)train_input_param --> train_input_handle
            paths, minibatch_size, input_data_type, is_output_sequence, name

    test_input_param: config of test dataset, it is set in a dict
    rain_input_param: config of train dataset, it is set in a dict
    '''
    if dataset_name not in datasets_map:
        raise ValueError('Name of DataSet unknown {}'.format(dataset_name))
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')

    if dataset_name == 'mnist':
        test_input_param = {
            'paths': valid_data_list,
            'minibatch_size': batch_size,
            'input_data_type': 'float32',
            'is_output_sequence': True,
            'name': dataset_name + 'Test Iterator'
        }
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param)#make config of test dataset
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {
                'paths': train_data_list,
                'minibatch_size': batch_size,
                'input_data_type': 'float32',
                'is_output_sequence': True,
                'name': dataset_name + 'Train Iterator'
            }
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param)#make config of train dataset
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
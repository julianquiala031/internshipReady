import json


def save_to_file(data, file_name):
    with open(file_name, 'w') as write_file:
        json.dump(data, write_file, indent=4)
        print('The file {0} was successfully created.'.format(file_name))


def read_from_file(file_name):
    with open(file_name, 'r') as read_file:
        data = json.load(read_file)
        print('You successfully read from {0}.'.format(file_name))
        return data

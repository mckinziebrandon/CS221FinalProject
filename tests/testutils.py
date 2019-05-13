from model_util import io_util


def get_resource_path(resource_name: str) -> str:
    resources_dir = io_util.Directory.of_file(__file__).subdir('resources')
    return resources_dir.get_file_path(resource_name, require_exists=True)


def get_resource_dir(dirname: str) -> io_util.Directory:
    resources_dir = io_util.Directory.of_file(__file__).subdir('resources')
    return resources_dir.subdir(dirname, require_exists=True)


def get_model_dir(dirname: str) -> io_util.Directory:
    base_models_dir = get_resource_dir('models')
    if dirname.endswith('*'):
        return io_util.Directory(base_models_dir.glob(dirname)[0])
    return base_models_dir.subdir(dirname)



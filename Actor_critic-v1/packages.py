def install_package(package_name):
        import pip
        from pip._internal import main as pipmain
        pipmain(['install', str(package_name)])


install_package('gym')
install_package('numpy')
install_package('os')
install_package('matplotlib')
install_package('torch')
install_package('pygame')
install_package('torchvision')

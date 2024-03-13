c = get_config()

c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.ServerApp.allow_remote_access = True
c.ServerApp.local_hostnames = ['localhost']
c.ServerApp.file_to_run = ''
c.ServerApp.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.AsyncMappingKernelManager'
c.ServerApp.root_dir = "workshops"
# c.ServerApp.password = ''
# c.ServerApp.password_required = False

#  Setting to an empty string disables authentication altogether, which is NOT
#  RECOMMENDED.
#  Default: '<generated>'
# c.ServerApp.token = '<generated>'

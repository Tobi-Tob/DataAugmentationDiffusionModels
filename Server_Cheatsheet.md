# Server Cheatsheet
This is a brief cheatsheet with useful commands to operate with the visinf Server.
## Basics
### Create SSH Session
1. activate VPN connection in WireGuard
2. Open bash / PowerShell and enter the follwoing command:
	```  
	ssh <username>@<labnumber>.visinf.informatik.tu-darmstadt.de
	```
	The labnumber is lab20, lab21 or lab22
### Other helpful commands
1. Change password with:
	```
	passwd
	```
	Follow the instructions
2. Exit SSH Session:
	```
	exit
	```
3. Get current directory:
	```
	pwd
	```
4. Create new directory:
	```
	mkdir <directory_path>
	```
	If directory_path starts with / the new directory is created relative to the root directory. Otherwise it is created relative to the current directory.
5. Change directory:
	```
	cd <directory_path>
	```
	Again, if directory_path starts with / the command cd searches relative to the root directory. Otherwise cd searches relative to the current directory. To jump to the parent directory you can use:
	```
	cd ..
	```
6. List all files and sub-directorys of the current directory:
	```
	ls
	```
7. Get a detailed list of all files and sub-directorys:
	```
	ls -al
	```
### tmux
tmux sessions can be used to run programs even if the ssh session is closed.

1. Check if tmux is installed:
	```
	tmux -V
	```
2. To create a new tmux session write:
	```
	tmux
	```
	If there is a green bar on the bottom of the console, the session was created correctly.
3. To detach the current tmux session press **ctrl + b** and then **d**. After doing so the ssh session can be closed without terminating the tmux session.
4. To re-attach to a tmux session write:
	```
	tmux attach
	```
5. To list all active tmux sessions write:
	```
	tmux ls
	```
7. To close a tmux session press **ctrl + b** and then **&**. The green bar on the bottom will turn yellow (this indicated that an input is expected) and by pressing **y** the tmux session is killed.

There are way more things to do with tmux. A video that explains the usage of tmux well is: https://www.youtube.com/watch?v=YYBQRBUMoFk
## Send files between local machine and server using SSH
To Send files between local machine and server using SSH you can use the **scp** command.
1. Send a file from local machine to the server:
	Open a Bash / Powershell at the origin directory on your local machine and type:
	```
	scp <file_name> <user_name>@<ip_adress>:<target_directory_on_server>
	```
	The *ip_adress* is something like *lab22.visinf.informatik.tu-darmstadt.de*. To copy the file to your user home directory the path is */visinf/home/username*
2. Send a whole directory from local machine to the server:
	```
	scp -r <local_path/dir_name> <user_name>@<ip_adress>:<target_directory_on_server>
 	scp -r D:\Uni\DLCV vilab07@lab22.visinf.informatik.tu-darmstadt.de:/visinf/home/vilab07/DataAugmentationDiffusionModels/synthetics
	```
3. Send a file from the server to the local machine:
	Open a Bash / Powershell **on your local machine** and navigate to the target directory and type:
	```
	scp <user_name>@<ip_adress>:<path_to_file_on_server/file_name>
	```
	The point at the end indicates that the current directory is the target directory on your local machine. Since you have navigated to the target directory before the filetransfer this works correctly.
4. Send a whole directory from the server to the local machine:
	Open a Bash / Powershell **on your local machine** and navigate to the target directory and type:
	```
	scp -r <user_name>@<ip_adress>:<path_to_file_on_server> <local_path_to_store>
 	scp -r vilab07@lab22.visinf.informatik.tu-darmstadt.de:/visinf/home/vilab07/DataAugmentationDiffusionModels/synthetics D:\Uni\DLCV
	```
	This works aquivalent to the above case.
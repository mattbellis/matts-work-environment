#!/bin/tcsh -f

set command = "ls"

set command = $command" -l"

echo $command
$command

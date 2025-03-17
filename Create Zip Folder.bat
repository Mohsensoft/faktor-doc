xcopy "build\html\*.*" "Z:\Help\Help_files\"
xcopy "build\html\_images\*.*" "Z:\Help\Help_files\_images\" /e /f /h
xcopy "build\html\_static\*.*" "Z:\Help\Help_files\_static\" /e /f /h
copy "Help.html" "Z:\Help\Help.html"
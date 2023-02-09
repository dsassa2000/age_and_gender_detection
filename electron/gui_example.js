const { execSync  } = require('child_process');
var nodeConsole = require('console');
var my_console = new nodeConsole.Console(process.stdout, process.stderr);

print_both('Initiating program');
function print_both(str) {
    console.log('Javascript: ' + str);
    my_console.log('Python: ' + str);
}

let files = [];
// starts program execution from within javascript and
function start_detect_function(evt) {
    print_both('start_detect_function');
    for (var i = 0; i < files.length; i++) {
        var myPic = files[i];
        execSync('python -u ./my_project/detection.py '+myPic.path+'');
    } 
    // var user = execSync('python -u ./my_project/detection.py C:\\Users\\HP\\Desktop\\gender_and_age_app\\gui\\my_project\\man1.jpg');
 }

 function file_browse_function(e) {
    files = e.target.files;
    console.log(files)
 }
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("start_detect").addEventListener("click", start_detect_function);
    document.getElementById("file_browse").addEventListener("change", file_browse_function);
});
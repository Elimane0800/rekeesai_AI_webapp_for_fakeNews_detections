let RunFakeNewsDetection = ()=>{
    sentence = document.getElementById("sentence").value;

    let xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            document.getElementById("system_response").innerHTML = xhttp.responseText;
        }
    };
    xhttp.open("GET", "fakenewsdetection?sentence"+"="+sentence, true);
    xhttp.send();
}
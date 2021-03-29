  function sendToFlask(apiRequest, data) {
  allData = {};
  allData.data = data;
  name = $('#myEx').dropdown('get value');
  console.log(apiRequest);


  var json = JSON.stringify(allData);
  var xhr = new XMLHttpRequest();
  currentTime = new Date();
  time = currentTime.getTime();
  var fileName = name + '-'+ String(time);
  xhr.open("PUT", 'api/' + name + '/'+ fileName, true);
  xhr.setRequestHeader('Content-type','application/json; charset=utf-8');
  xhr.onload = function () {
    var users = JSON.parse(xhr.responseText);
    if (xhr.readyState == 4 && xhr.status == "200") {
      console.log('Send to API', json);

    } else {
      console.error(' not ok');
    }
  }
  xhr.send(json);
}

function loadModels(){
  var xhr = new XMLHttpRequest();
  name = $('#myEx').dropdown('get value');
  console.log(name);
  name2 = $('#myEx').dropdown('get text');
  document.getElementById("loadName").innerHTML = `${name2} models are loaded`;

  xhr.open("GET", `api/loadModels/${name}`, true);
  xhr.setRequestHeader('Content-type','application/json; charset=utf-8');
  xhr.onload = function () {
    if (xhr.readyState == 4 && xhr.status == "200") {
      console.log('Send to API', xhr.responseText);

    } else {
      console.error(' not ok');
    }
  }
  xhr.send();
}

function trainModels(name){
  var xhr = new XMLHttpRequest();
  console.log(name);
  xhr.open("GET", `api/train/${name}`, true);
  xhr.setRequestHeader('Content-type','application/json; charset=utf-8');
  xhr.onload = function () {
    if (xhr.readyState == 4 && xhr.status == "200") {
      console.log('Send to API', xhr.responseText);

    } else {
      console.error(' not ok');
    }
  }
  xhr.send();
}


function clasifyEx(apiRequest, data){
  allData = {};
  allData.data = data;
  allData.tag = null;
  console.log(apiRequest, data);
  name = $('#myEx').dropdown('get value');
  var json = JSON.stringify(allData);
  var xhr = new XMLHttpRequest();
  currentTime = new Date();
  time = currentTime.getTime();
  var fileName = name+'-'+ String(time);
  xhr.open("PUT", apiRequest+name +'/'+fileName, true);
  xhr.setRequestHeader('Content-type','application/json; charset=utf-8');
  xhr.onload = function () {
    if (xhr.readyState == 4 && xhr.status == "200") {
      
      var allScores = JSON.parse(xhr.responseText);
      console.log(allScores.scores);
      document.getElementById('scoreG').innerHTML = `Score for good: ${allScores.scores.good}`
      document.getElementById('scoreB').innerHTML = `Score for bad: ${allScores.scores.bad}`
    } else {
      console.error(' not ok');
    }
  }
  xhr.send(json);

}



async function ttt(apiRequest){
  const res = await fetch(`${apiRequest}`);
  const json = await res.json();
  console.log(json);
}

function createEx(){
  //alert('test');
  //$('body').dimmer('show');
  document.getElementById("myForm").style.display = "block";
}

function hideIT(){
  //alert('test');
  //$('body').dimmer('show');
  document.getElementById("myForm").style.display = "none";
}


function makeEx(){
  var xhr = new XMLHttpRequest();
  var name = document.getElementById("newEx").value
  xhr.open("GET", `api/newExercise/${name}`, true);
  xhr.setRequestHeader('Content-type','application/json; charset=utf-8');
  xhr.onload = function () {
    if (xhr.readyState == 4 && xhr.status == "200") {
      location.reload(); 
      //alert('exercise created');

    } else {
      console.error(' not ok');
    }
  }
  xhr.send();
}
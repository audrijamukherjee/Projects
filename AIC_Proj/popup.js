// calls back with the url of the active browser tab
function getCurrentTabUrl(callback) {
  chrome.tabs.query({
    active: true,
    currentWindow: true
  }, function(tabs) {
    var tab = tabs[0];
    var url = tab.url;
    console.assert(typeof url == 'string', 'tab.url should be a string');
    callback(url);
  });
}
  
// requests score from server, passes response
function getScore(url, callback) {
  var x = new XMLHttpRequest();
  x.open('POST', 'http://localhost:8080?url=' + url);
  x.setRequestHeader("Content-type", "application/x-www-form-urlencoded")
  x.responseType = 'json';

  // onload handler - clears dots, passes results
  x.onload = function() {
    window.clearInterval(waiting);
    renderDots(0);
    if (x.status != 200 || !x.response) {
      return callback(x.status + ' - ' + x.statusText);
    }
    callback(null, x.response.score);
  };

  waiting = wait();
  x.send();
}

// sets the interval for changing dots 
function wait() {
  var numDots = 1;  
  return window.setInterval(function() {
    renderDots(numDots);
    if (numDots > 2) {
      numDots = 0;
    } else {
      numDots += 1;
    }
  }, 400);
}

// setter helpers
function renderMessage(messageText) {
  document.getElementById('message').innerHTML = messageText;
}
function renderDots(numDots) {
  document.getElementById('dots').innerHTML = '.'.repeat(numDots);
}

// main
document.addEventListener('DOMContentLoaded', function() {
  getCurrentTabUrl(function(url) {
  renderMessage('Getting fakeness score for active webpage');    
    getScore(url, function(err, score) {
      if (err) {
        renderMessage('Unable to get score: Error Code '+err);
      } else {
        score = Math.round(parseFloat(score)*100);
        renderMessage(
          'Fakeness: <font color="' + (score < 50 ? 'green':'red') + '">' + score + '</font>%'
        );
      }
    });
  });
});
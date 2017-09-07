var express = require('express');
var bodyParser = require('body-parser')
var spawn = require('child_process').spawn;

// using express app to handle requests
var app = express();
app.use(bodyParser.urlencoded({extended: true}));

// endpoint where the extension posts data
app.post('/', function (req, res) {

    (function(callback) {
        // gives up after 30 seconds 
        res.setTimeout(30000, function(){
            console.error('Request timed out');
            callback(408);
        });

        // python program runs as a child process
        getScore = spawn('python', ['create_train_test.py', req.query.url]);

        // stdout is the return
        getScore.stdout.on('data', function (data) {
            console.log('stdout: ' + data.toString());
            callback(null, data.toString());
        });
        // errors are reported
        getScore.stderr.on('data', function (data) {
            console.log('stderr: ' + data.toString());
            callback(500);
        });
        getScore.on('exit', function (code) {
            console.log('child process exited with code ' + code.toString());
        });
        
    })(function(errCode, score) {
        // access header for cross domain
        res.header("Access-Control-Allow-Origin", req.headers.origin);

        // if python fails signal internal server err
        if (errCode) {
            return res.sendStatus(errCode);
        }

        // responds to client with the scoring result
        res.status(200).json({ 'score': score });
        // could send back more info in future versions
    });
});

// run server
app.listen(8080, function () {
  console.log('listening on port 8080');
});

// setting up node working environment

var express  = require("express")
var app = express()
var bodyParser = require('body-parser')
var cors = require('cors')
var utils = require('./utils')()

PORT = process.env.PORT || 4000
app.use(bodyParser.json())

app.use(cors({
    origin: "http://localhost:4200"
}));


app.get('/api/', (req, res)=>{
	console.log("request hit on root")
	res.send(" weather API backend")
});

app.get('/api/analyze', (req, res)=>{

	console.log("Request hit on route 'analyze'") 
	utils.analyzeData().then((data)=>{

		// parse data back to obj
		parsed_res = JSON.parse(data)
		console.log(parsed_res)
		res.json(parsed_res)
	}, (err)=>{
		console.log(err)
		res.status(500).send(err)
	}).catch((err)=>{
		console.log(err)
		res.status(501).send(err)
	})
})


app.listen(PORT, ()=>{
	console.log("Server up on PORT :" + PORT)
});
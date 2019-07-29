module.exports = function(){

	return {

		analyzeData: ()=>{

			console.log("analyzeData() hit in utils")
			return new Promise((resolve, reject)=>{

				try{


				    const { spawn } = require('child_process');
				    const pyprog = spawn('python', ['./python-file.py', "Analysis"])
				    pyprog.stdout.on('data', function(data) {
				    	console.log("data came: ", data)
				        resolve(data.toString());
				    });

				    pyprog.stderr.on('data', (data) => {
				    	console.log("error came: ", data.toString())
				       	reject(data);    		    	
				    });
				}catch(ex){
					reject(ex)
				}
			});
		}
	}
}
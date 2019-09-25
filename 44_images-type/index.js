/***********************
 * @name JS
 * @author Jo.gel
 * @date 2019/9/25 0025
 ***********************/
let net;
async function app() {
	console.log("Loading mobilenet...")
	console.time("time")
	// 加载模型
	net =await mobilenet.load()
	console.info("success load model");
	
	// 通过我们图像上的模型进行预测。
	const imgEl =document.getElementById("img")
	const result =await net.classify(imgEl)
	console.timeEnd("time")
	console.info(result);
}

app()

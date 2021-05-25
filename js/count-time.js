/**
 * 计时器,每次重新计算效率有待优化
 * @param  {[type]} target [description]
 * @param  {[type]} d      [description]
 * @param  {[type]} h      [description]
 * @param  {[type]} m      [description]
 * @param  {[type]} s      [description]
 * @return {[type]}        [description]
 */
 var countTime = function (target, d, h, m, s) {
 	var cur = new Date().getTime();
 	var his = new Date(target).getTime();
 	diff = calcDifference(cur - his);
 	_countTime(d, h, m, s);
 	setInterval(function() {
 		_countTime(d, h, m, s);
 	}, 1000);
 }


/**
 * 更改显示值
 * @param  {[type]} d [description]
 * @param  {[type]} h [description]
 * @param  {[type]} m [description]
 * @param  {[type]} s [description]
 * @return {[type]}   [description]
 */
 var _countTime = function(d, h, m, s) {
 	$('#' + d).text(diff[0]);
 	$('#' + h).text(diff[1]);
 	$('#' + m).text(diff[2]);
 	$('#' + s).text(diff[3]);
 	diff[3] = (diff[3] + 1) % 60;
 	if (diff[3] == 0) {
 		diff[2] = (diff[2] + 1) % 60;
 		if (diff[2] == 0) {
 			diff[1] = (diff[1] + 1) % 24;
 			if (diff[1] == 0) {
 				diff[0] += 1;
 			}
 		}
 	}
 }


/**
 * 计算差距的具体时间
 * @param  {[type]} day [description]
 * @return {[type]}     [description]
 */
 var calcDifference = function(day) {
 	var days = Math.floor(day / (24 * 3600 * 1000));
 	var leave1 = day % (24 * 3600 * 1000);
 	var hours = Math.floor(leave1 / (3600 * 1000));
 	var leave2 = leave1 % (3600 * 1000);
 	var minutes = Math.floor(leave2 / (60 * 1000));
 	var leave3 = leave2 % (60 * 1000);
 	var seconds = Math.round(leave3 / 1000);
 	return new Array(days, hours, minutes, seconds);
 }

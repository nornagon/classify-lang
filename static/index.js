"use strict";
let langs = ['en', 'es', 'pt']
let langColors = {
  en: 'blue',
  es: 'green',
  pt: 'orange'
}

const debounce = function(func, wait, immediate) {
	var timeout;
	return function() {
		var context = this, args = arguments;
		var later = function() {
			timeout = null;
			if (!immediate) func.apply(context, args);
		};
		var callNow = immediate && !timeout;
		clearTimeout(timeout);
		timeout = setTimeout(later, wait);
		if (callNow) func.apply(context, args);
	};
};

let input = document.getElementById('text')
input.addEventListener('input', debounce((e => {
  let data = new FormData
  let sentence = e.target.value
  data.append('sentence', sentence)
  fetch('/classify', {method: 'POST', body: data}).then(res => res.json()).then(ret => {
    renderLangs(sentence, ret)
  })
}), 500))

function renderLangs(sentence, data) {
  let charwiseEstimates = []
  for (let line of data) {
    let [[begin, end], classes] = line
    end = Math.min(end, sentence.length)
    for (let i = begin; i < end; i++) {
      if (!charwiseEstimates[i]) {
        charwiseEstimates[i] = [[...classes], 1]
      } else {
        for (let j = 0; j < classes.length; j++) {
          charwiseEstimates[i][0][j] += classes[j]
        }
        charwiseEstimates[i][1] += 1
      }
    }
  }

  let output = document.getElementById('output')
  output.innerHTML = ''
  for (let i = 0; i < sentence.length; i++) {
    let letter = document.createElement('span')
    letter.textContent = sentence[i]
    let avgLangProb = charwiseEstimates[i][0].map(v => v / charwiseEstimates[i][1])
    let mostLikelyLang = langs[avgLangProb.indexOf(Math.max(...avgLangProb))]
    letter.style.color = langColors[mostLikelyLang]
    output.appendChild(letter)
  }
}

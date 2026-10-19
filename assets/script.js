(function(){
  var WA='23566000000'; // numéro WhatsApp (format international, sans +)
  var b=document.querySelector('.burger'),m=document.querySelector('nav ul');
  if(b)b.addEventListener('click',function(){var o=m.classList.toggle('open');b.setAttribute('aria-expanded',o)});
  if(m)m.addEventListener('click',function(e){if(e.target.tagName==='A')m.classList.remove('open')});
  var io='IntersectionObserver' in window?new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}})},{threshold:.12}):null;
  document.querySelectorAll('.rv').forEach(function(el){io?io.observe(el):el.classList.add('in')});
  var f=document.querySelectorAll('.filters button');
  f.forEach(function(btn){btn.addEventListener('click',function(){
    f.forEach(function(x){x.classList.remove('on')});btn.classList.add('on');
    document.querySelectorAll('.work').forEach(function(w){w.style.display=(btn.dataset.f==='all'||w.dataset.c===btn.dataset.f)?'':'none'});
  })});
  var form=document.getElementById('contact-form');
  if(form){
    var p=new URLSearchParams(location.search).get('offre');
    if(p&&form.service){for(var i=0;i<form.service.options.length;i++)if(form.service.options[i].value===p)form.service.selectedIndex=i}
    form.addEventListener('submit',function(e){
      e.preventDefault();
      if(form.website.value)return;
      var d=new FormData(form);
      var t='Bonjour Ndjam Digital, je suis '+d.get('nom')+'.\n'
        +'Projet : '+d.get('service')+'\nBudget : '+d.get('budget')+'\n'
        +'Téléphone : '+d.get('tel')+'\nEmail : '+d.get('email')+'\n\n'+d.get('message');
      document.getElementById('ok').style.display='block';
      window.open('https://wa.me/'+WA+'?text='+encodeURIComponent(t),'_blank','noopener');
    });
  }
})();

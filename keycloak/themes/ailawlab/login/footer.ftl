<#macro content>
<#-- Under every sign-in page: the terms acknowledgement, worded for what the page does. On the
     page where someone outside UW requests an account, first say what happens next, and send
     UW addresses to UW sign-on: an account made here with a UW address would stand in the way
     of that person's own UW sign-on later. -->
<#assign page = pageId!"">
<#if page == "register">
<div class="ailawlab-register-note">
  <p><strong>For people outside the University of Wyoming.</strong> After you create the account
    you will be asked who you are and what you would like to use the lab for. An administrator
    reviews each request; until it is approved the account cannot be used.</p>
  <p>At UW? Don't create an account here: <a href="${url.loginUrl}">go back and use UW sign-on</a>.</p>
</div>
<script>
  // UW addresses belong to UW sign-on (checked again by the administrators who approve requests).
  (function () {
    var form = document.getElementById("kc-register-form");
    var email = document.getElementById("email");
    var note = document.querySelector(".ailawlab-register-note");
    if (form && note) { form.parentNode.insertBefore(note, form); note.classList.add("above-form"); }
    if (!form || !email) return;
    function uw() { return /@(.+\.)?uwyo\.edu\s*$/i.test(email.value); }
    function check() {
      email.setCustomValidity(uw() ? "This is a UW address: please go back and use UW sign-on instead." : "");
    }
    email.addEventListener("input", check);
    form.addEventListener("submit", function (e) {
      check();
      if (uw()) { e.preventDefault(); email.reportValidity(); }
    });
  })();
</script>
</#if>
<p class="ailawlab-ack">By <#if page == "register">creating an account<#else>signing in</#if>, I acknowledge
  that I have read and agree to the
  <a href="/ai_law_lab/#terms" target="_blank" rel="noopener">AI Law Lab Terms of Use</a>. I understand
  that the AI Law Lab is an experimental academic research environment; that its outputs do not
  constitute legal advice; that I am responsible for ensuring lawful and ethical use of any materials
  I upload; that I must not upload confidential, privileged, personally identifiable, or restricted
  information; and that I am responsible for independently verifying all generated outputs.</p>
</#macro>

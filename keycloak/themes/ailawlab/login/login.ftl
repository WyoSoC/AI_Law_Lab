<#-- The AI Law Lab's sign-in page: Keycloak's own (keycloak.v2 login.ftl), reordered. UW
     sign-on comes first, as the way most people sign in; an AI Law Lab account (email and
     password) below it; then, for people outside UW, a link to request an account, which an
     admin approves. The terms acknowledgement follows in footer.ftl. -->
<#import "template.ftl" as layout>
<#import "field.ftl" as field>
<#import "buttons.ftl" as buttons>
<#import "passkeys.ftl" as passkeys>
<@layout.registrationLayout displayMessage=!messagesPerField.existsError('username','password') displayInfo=false; section>
<!-- template: login.ftl (ailawlab) -->

    <#if section = "header">
        ${msg("loginAccountTitle")}
    <#elseif section = "form">
        <#assign providers = (social.providers)![]>
        <#assign uw = providers?filter(p -> p.alias == "uwyo")>
        <#assign others = providers?filter(p -> p.alias != "uwyo")>
        <#if uw?has_content>
            <div class="ailawlab-uw">
                <a id="social-uwyo" data-once-link class="${properties.kcButtonPrimaryClass!} ${properties.kcButtonBlockClass!} ailawlab-uw-button"
                   href="${uw[0].loginUrl}">Sign in with UW sign-on</a>
                <p class="ailawlab-uw-note">For University of Wyoming faculty, staff and students, with your UW username and password.</p>
            </div>
        </#if>
        <#if realm.password>
            <#if uw?has_content || others?has_content>
                <div class="ailawlab-divider" role="separator"><span>Or sign in with an AI Law Lab account</span></div>
            </#if>
            <div id="kc-form">
              <div id="kc-form-wrapper">
                <form id="kc-form-login" class="${properties.kcFormClass!}" onsubmit="login.disabled = true; return true;" action="${url.loginAction}" method="post" novalidate="novalidate">
                    <#if !usernameHidden??>
                        <#assign label>
                            <#if !realm.loginWithEmailAllowed>${msg("username")}<#elseif !realm.registrationEmailAsUsername>${msg("usernameOrEmail")}<#else>${msg("email")}</#if>
                        </#assign>
                        <@field.input name="username" label=label error=messagesPerField.getFirstError('username','password')
                            autofocus=!uw?has_content autocomplete="${(enableWebAuthnConditionalUI?has_content)?then('username webauthn', 'username')}" value=login.username!'' />
                        <@field.password name="password" label=msg("password") error="" forgotPassword=realm.resetPasswordAllowed autofocus=usernameHidden?? autocomplete="current-password">
                            <#if realm.rememberMe && !usernameHidden??>
                                <@field.checkbox name="rememberMe" label=msg("rememberMe") value=login.rememberMe?? />
                            </#if>
                        </@field.password>
                    <#else>
                        <@field.password name="password" label=msg("password") forgotPassword=realm.resetPasswordAllowed autofocus=usernameHidden?? autocomplete="current-password">
                            <#if realm.rememberMe && !usernameHidden??>
                                <@field.checkbox name="rememberMe" label=msg("rememberMe") value=login.rememberMe?? />
                            </#if>
                        </@field.password>
                    </#if>
                    <input type="hidden" id="id-hidden-input" name="credentialId" <#if auth.selectedCredential?has_content>value="${auth.selectedCredential}"</#if>/>
                    <@buttons.actionGroup>
                        <button class="${properties.kcButtonSecondaryClass!} ${properties.kcButtonBlockClass!}" name="login" id="kc-login" type="submit">${msg("doLogIn")}</button>
                    </@buttons.actionGroup>
                </form>
              </div>
            </div>
            <@passkeys.conditionalUIData />
        </#if>
        <#if others?has_content>
            <ul class="ailawlab-other-providers">
                <#list others as p>
                    <li><a id="social-${p.alias}" data-once-link class="${properties.kcButtonSecondaryClass!} ${properties.kcButtonBlockClass!}"
                           href="${p.loginUrl}">Sign in with ${p.displayName!}</a></li>
                </#list>
            </ul>
        </#if>
        <#if realm.password && realm.registrationAllowed && !registrationDisabled??>
            <p class="ailawlab-request">Not at UW? <a id="ailawlab-request-link" href="${url.registrationUrl}">Request an account</a>.
              An administrator reviews each request before the account can be used.</p>
        </#if>
    </#if>

</@layout.registrationLayout>

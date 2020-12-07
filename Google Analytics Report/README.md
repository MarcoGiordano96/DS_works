<h1>Overview</h1>
<p>Google Analytics is one of the most preferred free solutions by Digital Marketers and SEO Specialists to keep track of results and analyze details about visitors of your website(s). Despite its popularity, not all the professionals know about its API (Google Analytics Reporting API v4) and how we could use it to automate reports, instead of relying on slower tools or doing everything by hand!</p>

<h2>The setup: what you need to do before we start</h2>
<p>What I'll cover in this section is pretty easy if you follow all the steps. I admit that the first time I had some difficulties due to Google's documentation and my inexperience with APIs. Once you get used to it, it becomes very easy and repeatable.
  
Before we start, I assume you already have a website running Google Analytics. If you have one, very well, we can continue and start our setup.

1) Create a project Google API Console via the <a href = 'https://console.developers.google.com/flows/enableapi?apiid=analyticsreporting.googleapis.com&credential=client_key'> setup tool</a>. Follow the instructions to get a JSON file, which you will rename to client_secrets.json. Please be careful and store it in a safe place, as it contains the data necessary to access your GA API account. Moreover, you should copy the service email, which always starts with project.
2) The most difficult part has gone, now you have to visit Google Analytics and open the Admin tab from the property you are interested in. Go to User Management and grant your service account permission (the email you copied before!), Read & Analyze will do the trick. 
3) Finally, copy View ID of the view you are interested in. 

#to continue </p>

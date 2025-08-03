from playwright.sync_api import sync_playwright
import time
password = "Sertaindeat1"








with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto('http://e-fich.unl.edu.ar/moodle/entrar.php')
    page.get_by_placeholder("USUARIO").fill("mateo.centis@hotmail.com.ar")
    page.get_by_placeholder("****").fill(password)
    page.get_by_role("button").click()

    page.pause()
    # main()
    # time.sleep(5)
    # browser.close() 

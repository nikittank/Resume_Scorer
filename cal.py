import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import unittest


class CalculatorTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            options.add_argument("--no-sandbox")  # Added for stability
            options.add_argument("--disable-dev-shm-usage")  # Added for stability
            cls.driver = webdriver.Chrome(options=options)
            cls.driver.implicitly_wait(10)  # Added implicit wait
            cls.wait = WebDriverWait(cls.driver, 15)
            cls.driver.get("https://www.calculator.net/")
            time.sleep(3)
            cls._accept_cookies()
            cls._clear_calculator()
        except Exception as e:
            print(f"Failed to initialize WebDriver: {str(e)}")
            raise

    @classmethod
    def _accept_cookies(cls):
        """Accept cookies if popup appears"""
        try:
            accept_button = cls.wait.until(
                EC.element_to_be_clickable((By.ID, "cookieconsentallowall"))
            )
            accept_button.click()
            time.sleep(1)
        except Exception:
            pass  # Cookies popup might not be present

    @classmethod
    def _clear_calculator(cls):
        """Clear calculator with multiple methods"""
        try:
            # Try to clear using C button
            cls.driver.find_element(By.ID, "sciInPut06").click()
            # Try to clear using AC button
            cls.driver.find_element(By.ID, "sciInPut18").click()
        except Exception:
            try:
                # Alternative clear method
                cls.driver.find_element(By.CSS_SELECTOR, ".clearbtn").click()
            except Exception:
                pass  # Continue if clear fails
        time.sleep(0.3)

    @classmethod
    def _is_browser_alive(cls):
        """Check if the browser window is still open"""
        try:
            # Try to get current URL - if this fails, browser is closed
            cls.driver.current_url
            return True
        except Exception:
            return False

    def _click_button(self, button_text):
        """Universal button clicker with improved reliability"""
        button_id_map = {
            "0": "sciInPut07", "1": "sciInPut08", "2": "sciInPut09",
            "3": "sciInPut10", "4": "sciInPut11", "5": "sciInPut12",
            "6": "sciInPut13", "7": "sciInPut14", "8": "sciInPut15",
            "9": "sciInPut16",
            "+": "sciInPut02", "-": "sciInPut03", "*": "sciInPut04",
            "/": "sciInPut05", "=": "sciInPut01",
            "C": "sciInPut06", ".": "sciInPut17", "AC": "sciInPut18"
        }
        
        try:
            # First check if browser is still open
            if not self._is_browser_alive():
                raise Exception("Browser window is closed")
            
            # Try by ID first
            if button_text in button_id_map:
                button = self.wait.until(
                    EC.element_to_be_clickable((By.ID, button_id_map[button_text]))
                )
                button.click()
                time.sleep(0.2)
                return
            
            # Fallback to other methods
            button_map = {"*": "×", "/": "÷"}
            text_to_try = button_map.get(button_text, button_text)
            
            locators = [
                (By.XPATH, f"//span[text()='{text_to_try}']"),
                (By.XPATH, f"//div[text()='{text_to_try}']"),
                (By.CSS_SELECTOR, f"span[onclick*='{text_to_try.lower()}']"),
                (By.CSS_SELECTOR, f"input[value='{text_to_try}']")
            ]
            
            for locator in locators:
                try:
                    button = self.wait.until(EC.element_to_be_clickable(locator))
                    button.click()
                    time.sleep(0.2)
                    return
                except Exception:
                    continue
            
            raise Exception(f"Button '{button_text}' not found")
            
        except Exception as e:
            if self._is_browser_alive():
                try:
                    self.driver.save_screenshot(f"button_error_{button_text}.png")
                except Exception:
                    pass
            raise Exception(f"Failed to click '{button_text}': {str(e)}")

    def _is_browser_alive(self):
        """Check if the browser window is still open"""
        try:
            # Try to get current URL - if this fails, browser is closed
            self.driver.current_url
            return True
        except Exception:
            return False

    def _get_result(self):
        """Get calculator display result with multiple fallbacks"""
        try:
            if not self._is_browser_alive():
                raise Exception("Browser window is closed")
            
            result = self.wait.until(
                EC.visibility_of_element_located((By.ID, "sciOutPut"))
            ).text.strip()
            return result.replace('\n', '').replace(' ', '')
        except Exception:
            try:
                return self.driver.find_element(By.CSS_SELECTOR, ".sciout").text.strip()
            except Exception:
                if self._is_browser_alive():
                    self.driver.save_screenshot("result_error.png")
                raise Exception("Could not get calculator result")

    def _clear_calculator(self):
        """Clear calculator with multiple methods"""
        try:
            # Try to clear using C button
            self.driver.find_element(By.ID, "sciInPut06").click()
            # Try to clear using AC button
            self.driver.find_element(By.ID, "sciInPut18").click()
        except Exception:
            try:
                # Alternative clear method
                self.driver.find_element(By.CSS_SELECTOR, ".clearbtn").click()
            except Exception:
                pass  # Continue if clear fails
        time.sleep(0.3)

    def test_addition(self):
        """Test addition: 7 + 3 = 10"""
        self._click_button("7")
        self._click_button("+")
        self._click_button("3")
        self._click_button("=")
        self.assertEqual(self._get_result(), "10", "7 + 3 should equal 10")
        self._clear_calculator()

    def test_multiplication(self):
        """Test multiplication: 5 × 6 = 30"""
        self._click_button("5")
        self._click_button("*")
        self._click_button("6")
        self._click_button("=")
        self.assertEqual(self._get_result(), "30", "5 × 6 should equal 30")
        self._clear_calculator()

    def test_division(self):
        """Test division: 8 ÷ 4 = 2"""
        self._click_button("8")
        self._click_button("/")
        self._click_button("4")
        self._click_button("=")
        self.assertEqual(self._get_result(), "2", "8 ÷ 4 should equal 2")
        self._clear_calculator()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'driver') and cls._is_browser_alive():
            cls.driver.quit()


if __name__ == "__main__":
    unittest.main()
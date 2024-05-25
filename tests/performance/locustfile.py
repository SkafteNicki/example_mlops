import os
import random

from locust import HttpUser, between, task

_PROJECT_ROOT = os.getcwd()  # assume we always run pytest from the project root
_TEST_ROOT = os.path.join(_PROJECT_ROOT, "tests")  # root of test folder

test_imgs = {
    1: os.path.join(_TEST_ROOT, "img_1.jpg"),
    2: os.path.join(_TEST_ROOT, "img_2.jpg"),
    3: os.path.join(_TEST_ROOT, "img_3.jpg"),
}


class MnistClassifierUser(HttpUser):
    """A user class for the MNIST classifier API."""

    wait_time = between(1, 2.5)

    @task
    def index(self):
        """A task to hit the index route."""
        self.client.get("/")

    @task(2)
    def health(self):
        """A task to hit the health route."""
        self.client.get("/health")

    @task(3)
    def predict(self):
        """A task to hit the predict route."""
        test_img = test_imgs[random.randint(1, 3)]
        self.client.post("/predict", json={"image": test_img})

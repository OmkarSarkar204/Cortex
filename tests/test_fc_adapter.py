import unittest
from unittest.mock import MagicMock
from cortex.fc_adapter import FCAdapter

class TestFCAdapter(unittest.TestCase):

    def test_initialization(self):
        adapter = FCAdapter("COM3", 115200)
        self.assertEqual(adapter.connection_string, "COM3")
        self.assertEqual(adapter.baud, 115200)

    def test_send_land_without_connection(self):
        adapter = FCAdapter("COM3", 115200)
        adapter.master = MagicMock()
        adapter.master.target_system = 1
        adapter.master.target_component = 1
        adapter.master.mav = MagicMock()

        adapter.send_land()

        adapter.master.mav.command_long_send.assert_called()


if __name__ == "__main__":
    unittest.main()
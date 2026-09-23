import asyncio
import os
import socket
import traceback

from uni_agent.sandbox.cube import CubeSandbox


TEMPLATE = "tpl-a4a7b3286f5e4ad6966f276e"
API_URL = "http://174.1.59.1:3000"


async def main():
    print("python:", os.sys.executable)
    print("template:", TEMPLATE)
    print("api_url:", API_URL)

    try:
        print("resolving API host...")
        print(socket.gethostbyname("174.1.59.1"))
    except Exception:
        traceback.print_exc()

    sb = CubeSandbox(
        template=TEMPLATE,
        api_url=API_URL,
    )

    try:
        print("starting sandbox...")
        await sb.start()
        print("sandbox started")

        for argv in (
            ["echo", "hello"],
            ["pwd"],
            ["bash", "-c", "echo hello"],
            ["python3", "-c", "print('hello')"],
        ):
            print("\nexecuting:", argv)
            try:
                result = await sb.exec(argv, timeout=30)
                print("exit_code:", result.exit_code)
                print("stdout:", repr(result.stdout))
                print("stderr:", repr(result.stderr))
            except Exception:
                print("exec exception:")
                traceback.print_exc()

    except Exception:
        print("sandbox startup exception:")
        traceback.print_exc()

    finally:
        print("\nclosing sandbox...")
        try:
            await sb.stop()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
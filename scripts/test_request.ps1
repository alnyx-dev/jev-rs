$ErrorActionPreference = "Stop"

$payload = @{
    state = "Customer: I was charged twice for my subscription, but my order still shows Pending. Please fix this or issue a refund immediately!"
    questions = @{
        is_urgent = @{
            type = "noul"
            instructions = "Does this require urgent human intervention?"
        }
        category = @{
            type = "choice"
            instructions = "Select issue category"
            criteria = @{
                billing = "Payment, invoice, or refund issues"
                tech_bug = "App crash or visual error"
                delivery = "Shipping and package tracking"
            }
        }
        frustration = @{
            type = "score"
            instructions = "Customer frustration level"
            criteria = @("Calm", "Annoyed", "Extremely Furious")
        }
    }
} | ConvertTo-Json -Depth 5

$bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)

Write-Host "[*] Sending request to http://127.0.0.1:8000/v1/systemone..." -ForegroundColor Cyan

$res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/systemone" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body $bytes

$res | ConvertTo-Json -Depth 5